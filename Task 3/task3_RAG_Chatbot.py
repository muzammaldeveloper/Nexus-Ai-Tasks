import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re


knowledge_base_text = """
GlobalMart Product Information

Electronics Category:
- SmartHome Pro Hub: A centralized smart home controller compatible with 200+ devices. Price: $129.99. Connectivity: WiFi, Bluetooth, Zigbee.
- UltraView 4K TV: 65-inch 4K OLED TV with HDR10+ and Dolby Atmos. Resolution: 3840x2160. Refresh rate: 120Hz. Smart features: Built-in Google TV.
- SoundMax Bluetooth Speaker: Waterproof portable speaker with 20-hour battery life. Output power: 20W. Connectivity: Bluetooth 5.0, AUX-in.

Home & Kitchen Category:
- ChefPro Air Fryer: Digital air fryer with 5.5L capacity and 7 preset cooking functions. Power: 1700W. Temperature range: 180-400°F.
- ComfortSleep Memory Foam Pillow: Orthopedic pillow with cooling gel layer. Dimensions: 24x16 inches. Material: CertiPUR-US certified foam.

Sports & Outdoors Category:
- TrailMaster Hiking Backpack: 40L waterproof backpack with adjustable suspension system. Material: Ripstop nylon. Features: Multiple compartments, hydration sleeve.
- FlexFlow Yoga Mat: Eco-friendly non-slip mat with alignment markers. Thickness: 6mm. Size: 72x24 inches. Material: Natural rubber.

Health & Beauty Category:
- GlowMax LED Therapy Mask: Professional-grade LED mask for skin treatment. LEDs: 150 (Red & Blue). Treatment time: 10-30 minutes per session.
- SonicClean Toothbrush: Electric toothbrush with 4 cleaning modes and wireless charging. Battery life: 3 weeks. Included: 4 brush heads.

Customer Service Information:
- Returns Policy: 30-day return window for unused products in original packaging.
- Shipping: Free standard shipping on orders over $50. Express shipping available.
- Warranty: Standard 1-year manufacturer warranty on all electronics. Extended warranties available.
"""

class RAGSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        
    def prepare_knowledge_base(self, text):
        """Split the text into chunks and create embeddings"""
        
        chunks = re.split(r'\n\s*\n', text)
        self.knowledge_base = [chunk.strip() for chunk in chunks if chunk.strip()]
        
       
        self.embeddings = self.model.encode(self.knowledge_base, normalize_embeddings=True)
        
       
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index (equivalent to cosine similarity with normalized vectors)
        self.index.add(self.embeddings)
        
    def search(self, query, top_k=3):
        """Search for the most relevant facts for a query"""
        if self.index is None or len(self.knowledge_base) == 0:
            return ["Knowledge base not initialized. Please prepare the knowledge base first."]
        
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
       
        distances, indices = self.index.search(query_embedding, top_k)
        
       
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.knowledge_base):
                results.append({
                    'fact': self.knowledge_base[idx],
                    'score': distances[0][i]
                })
        
        return results

@st.cache_resource
def initialize_rag_system():
    rag = RAGSystem()
    rag.prepare_knowledge_base(knowledge_base_text)
    return rag

# Streamlit UI
def main():
    st.set_page_config(page_title="GlobalMart RAG System", page_icon="🔍", layout="wide")
    
    st.title("🔍 GlobalMart Product Information Retrieval System")
    st.markdown("This system uses Retrieval-Augmented Generation (RAG) to find relevant product information from GlobalMart's knowledge base.")
    
    
    rag_system = initialize_rag_system()
    
   
    query = st.text_input("Enter your question about GlobalMart products:", 
                         placeholder="E.g., What smart home products do you offer?")
    
    
    top_k = st.slider("Number of results to show:", min_value=1, max_value=5, value=3)
    
    if st.button("Search") and query:
        with st.spinner("Searching for relevant information..."):
            results = rag_system.search(query, top_k=top_k)
            
        st.subheader("Top Results:")
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1} (Relevance: {result['score']:.4f})"):
                st.write(result['fact'])
    
   
    with st.expander("View Complete Knowledge Base"):
        st.text(knowledge_base_text)

if __name__ == "__main__":
    main()