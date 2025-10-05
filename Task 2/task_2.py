import spacy
from spacy.matcher import Matcher
from spacy import displacy
from spacy.tokens import Span
import streamlit as st

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Create Matcher for custom ORDER_ID
matcher = Matcher(nlp.vocab)
order_pattern = [
    [{"TEXT": {"REGEX": r"#\d+"}}],
    [{"LOWER": "order"}, {"IS_DIGIT": True}]
]
matcher.add("ORDER_ID", order_pattern)

# Function to extract entities
def extract_entities(text):
    doc = nlp(text)
    matches = matcher(doc)

    ents = list(doc.ents)  
    for match_id, start, end in matches:
        span = Span(doc, start, end, label=nlp.vocab.strings["ORDER_ID"])  
        ents.append(span)

    doc.ents = ents  
    return doc


st.title("Customer Query Entity Extractor (NER)")
st.write("Type any query (e.g., 'Where is my order #12345?') and I will extract key information.")


query = st.text_area("Enter your query:")

if st.button("Extract Entities"):
    if query.strip() != "":
        doc = extract_entities(query)

        st.subheader("🔎 Extracted Entities:")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"- **{ent.text}** ({ent.label_})")
        else:
            st.write("No entities found.")

        # Optional: displaCy visualization
        html = displacy.render(doc, style="ent", jupyter=False)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a query first.")
