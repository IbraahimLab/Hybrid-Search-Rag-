
import gradio as gr
import numpy as np
import torch
from pypdf import PdfReader
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import time # Import time for simulation of typing (UX improvement)

# --- 1. CONFIGURATION ---

# IMPORTANT: Replace with your actual Groq API Key.
# Best practice is loading this from an environment variable
GROQ_API_KEY = None

# Initialize the Groq client
try:
    # Using the currently supported model for Llama 3 8B
    LLM_MODEL = "llama-3.1-8b-instant" 
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    print("Please ensure your GROQ_API_KEY is correct.")
    groq_client = None


# --- 2. RETRIEVER SETUP (Same as before) ---

# Global variables to store documents and models
documents = []
embed_model = None
dense_embeddings = None
bm25 = None
reranker = None

# Function to load and initialize models/data
def setup_rag():
    global documents, embed_model, dense_embeddings, bm25, reranker
    
    pdf_files = ["how to write ml paper.pdf"] 
    CHUNK_SIZE = 500
    
    print("Loading documents and chunking...")

    for file in pdf_files:
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            documents.extend(chunks)
        except FileNotFoundError:
            print(f"Error: PDF file not found at {file}. Please check the path.")
            return False
        except Exception as e:
            print(f"Error processing PDF {file}: {e}")
            return False

    if not documents:
        print("FATAL: No documents loaded.")
        return False
        
    print(f"Loaded {len(documents)} document chunks.")
    print("Initializing embedding models...")

    # Dense Embeddings Model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dense_embeddings = embed_model.encode(documents, convert_to_tensor=True).cpu()

    # Sparse Vectors (BM25)
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    # Cross-Encoder for Reranking
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    return True

# --- 3. HYBRID SEARCH AND RERANKING LOGIC (Same as before) ---

def reciprocal_rank_fusion(dense_scores, bm25_scores, k=60):
    # RRF implementation from previous response
    dense_ranks_map = {doc_idx: rank + 1 for rank, doc_idx in enumerate(np.argsort(-dense_scores))}
    bm25_ranks_map = {doc_idx: rank + 1 for rank, doc_idx in enumerate(np.argsort(-bm25_scores))}
    
    fusion_scores = {}
    for i in range(len(dense_scores)): 
        rank_dense = dense_ranks_map.get(i, len(dense_scores) + 1)
        rank_bm25 = bm25_ranks_map.get(i, len(dense_scores) + 1)
        score = (1 / (k + rank_dense)) + (1 / (k + rank_bm25))
        fusion_scores[i] = score
        
    combined_idx = sorted(fusion_scores, key=fusion_scores.get, reverse=True)
    return combined_idx


def rerank(query, candidate_docs):
    pairs = [[query, doc] for doc in candidate_docs]
    scores = reranker.predict(pairs)
    ranked_idx = np.argsort(-scores)
    return [candidate_docs[i] for i in ranked_idx]


def hybrid_search(query, top_candidates=10, final_top_k=5):
    """Performs hybrid retrieval, RRF fusion, and final reranking."""
    
    # Check if RAG components are loaded
    if not all([embed_model, bm25, reranker]):
        return ["RAG components not loaded. Check setup_rag()."]

    # 1. Dense Retrieval (Cosine Similarity)
    query_emb = embed_model.encode([query], convert_to_tensor=True).cpu()
    cosine_scores = torch.nn.functional.cosine_similarity(query_emb, dense_embeddings).cpu().numpy()
    
    # 2. Sparse Retrieval (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 3. RRF Fusion
    combined_idx = reciprocal_rank_fusion(cosine_scores, bm25_scores)
    
    # Take top candidates for reranker
    top_candidates_docs = [documents[i] for i in combined_idx[:top_candidates]]
    
    # 4. Rerank
    reranked = rerank(query, top_candidates_docs)
    
    return reranked[:final_top_k]


# --- 4. LLM COMMUNICATION (Updated Model) ---

def ask_groq(query, context_docs, model_name):
    """Communicates with the Groq API for chat completion."""
    if not groq_client:
        return "ERROR: Groq client is not initialized. Check your API key.", context_docs
        
    context_text = "\n".join(context_docs)
    
    system_prompt = (
        "You are an expert RAG assistant specializing in the provided documents. "
        "Answer the user's question based ONLY on the context provided below. "
        "If the context does not contain the answer, state clearly that you cannot find the answer in the documents. "
        "Maintain a helpful and professional tone."
    )
    
    user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}"
    
    try:
        chat_completion = groq_client.chat.completions.create(
            model=model_name, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Add temperature setting for control
            temperature=0.1
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {e}. Please check the console.", context_docs


# --- 5. GRADIO HANDLER FUNCTION ---

def user(user_message, chat_history):
    # Append the user's message to the chat history
    return "", chat_history + [[user_message, None]]

def bot(chat_history, rag_params, model_name):
    user_message = chat_history[-1][0]
    
    # 1. Extract RAG parameters from the state (a list of values from the sliders)
    # [top_candidates, final_top_k, rrf_k]
    top_candidates, final_top_k, rrf_k = rag_params
    
    # 2. Hybrid search + rerank
    retrieved_docs = hybrid_search(user_message, top_candidates=top_candidates, final_top_k=final_top_k)
    
    # Format context for display
    context_for_ui = [f"Chunk {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    context_str = "\n\n---\n\n".join(context_for_ui)

    # 3. LLM answer generation
    answer = ask_groq(user_message, retrieved_docs, model_name)
    
    # Simulate typing delay for better UX
    # for char in answer:
    #     chat_history[-1][1] = (chat_history[-1][1] or "") + char
    #     yield chat_history, context_str
    #     time.sleep(0.01)

    # Replace the last turn placeholder with the actual answer
    chat_history[-1][1] = answer
    
    # Return updated chat history and the context string for the readout panel
    return chat_history, context_str


def clear_all():
    return None, None, [10, 5, 60], LLM_MODEL # Reset chat, context, and parameters


# --- 6. GRADIO INTERFACE (Blocks) ---

if setup_rag():
    with gr.Blocks(title="Hybrid RAG Chatbot (Groq + RRF)") as demo:
        # State to store RAG parameters (top_candidates, final_top_k, rrf_k)
        rag_params = gr.State([10, 5, 60])
        model_state = gr.State(LLM_MODEL)

        gr.Markdown("# 🧠 Hybrid RAG Chatbot (Groq Engine)")
        gr.Markdown("This prototype uses BM25 and Dense Embeddings combined via Reciprocal Rank Fusion (RRF), followed by a MiniLM Reranker, to provide context for a Groq LLM (Llama 3.1 8B).")

        with gr.Row():
            # --- SIDEBAR (Settings) ---
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ RAG Settings")
                
                # Model Selector
                model_dropdown = gr.Dropdown(
                    label="Groq LLM Model",
                    choices=["llama-3.1-8b-instant", "llama3-70b-8192"],
                    value=LLM_MODEL,
                    interactive=True
                )
                
                gr.Markdown("### Retrieval Parameters")
                
                # Sliders for easy parameter tuning
                slider_candidates = gr.Slider(
                    minimum=5, maximum=50, step=1, value=10, 
                    label="Top Candidates for Reranker (Hybrid)", 
                    info="Number of chunks retrieved before final reranking."
                )
                
                slider_reranked = gr.Slider(
                    minimum=1, maximum=10, step=1, value=5, 
                    label="Final Context Chunks (to LLM)",
                    info="Number of final chunks passed to the LLM as context."
                )
                
                slider_rrf_k = gr.Slider(
                    minimum=1, maximum=100, step=5, value=60, 
                    label="RRF Constant (k)", 
                    info="Fusion constant: higher k smooths rank differences."
                )
                
                # Combine slider outputs into the state
                gr.Button("Apply RAG Settings", variant="primary").click(
                    fn=lambda a, b, c, d: [[a, b, c], d],
                    inputs=[slider_candidates, slider_reranked, slider_rrf_k, model_dropdown],
                    outputs=[rag_params, model_state]
                )

                clear_btn = gr.Button("🗑️ Clear Chat and Reset Settings", variant="secondary")

            # --- MAIN CHAT AREA ---
            with gr.Column(scale=3):
                # Chat component
                chatbot = gr.Chatbot(
                    height=500,
                    avatar_images=[None, "https://storage.googleapis.com/groq-marketing/images/Groq_Icon_Color.png"] # Use a Groq icon if possible
                )
                
                # Message input row
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question", 
                        placeholder="Ask a question about the documents...", 
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            # --- CONTEXT READOUT PANEL ---
            with gr.Column(scale=2):
                gr.Markdown("## 🔍 Retrieved Context Readout")
                context_output = gr.Textbox(
                    label=f"Top Chunks Passed to {LLM_MODEL}", 
                    lines=20, 
                    max_lines=20, 
                    interactive=False, 
                    info="The final, reranked documents used by the LLM to formulate its answer."
                )   

        # --- EVENT HANDLING ---
        
        # User submission triggers two steps:
        # 1. Update chat history with user message
        # 2. Get bot response
        
        submit_btn.click(
            user, 
            [msg, chatbot], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot, 
            [chatbot, rag_params, model_state], 
            [chatbot, context_output]
        )
        
        # Enable submitting via Enter key
        msg.submit(
            user, 
            [msg, chatbot], 
            [msg, chatbot], 
            queue=False
        ).then(
            bot, 
            [chatbot, rag_params, model_state], 
            [chatbot, context_output]
        )

        # Clear button logic
        clear_btn.click(
            clear_all, 
            None, 
            [chatbot, context_output, rag_params, model_state], 
            queue=False
        )

    demo.launch()

else:
    print("RAG setup failed. Cannot launch Gradio interface.")