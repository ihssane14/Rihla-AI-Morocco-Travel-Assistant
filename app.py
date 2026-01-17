
import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Rihla - Morocco Travel Guide",
    page_icon="🇲🇦",
    layout="centered"
)

# Professional CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(180deg, #e0f2f1 0%, #b2dfdb 100%);
}

#MainMenu, footer, header {visibility: hidden;}

.main .block-container {
    padding: 1.5rem 1rem 4rem 1rem;
    max-width: 950px;
}

/* Compact Header */
.compact-header {
    background: white;
    border-radius: 20px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.logo-circle {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #26a69a, #00897b);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 3px 10px rgba(38,166,154,0.25);
}

.brand-info h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00695c;
}

.brand-info p {
    margin: 0.2rem 0 0 0;
    font-size: 0.8rem;
    color: #546e7a;
    font-weight: 500;
}

.ai-badge {
    background: linear-gradient(135deg, #e0f2f1, #b2dfdb);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #00695c;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pending_query' not in st.session_state:
    st.session_state.pending_query = None

if 'show_itinerary_form' not in st.session_state:
    st.session_state.show_itinerary_form = False

# Header
st.markdown("""
<div class="compact-header">
    <div class="header-left">
        <div class="logo-circle">🇲🇦</div>
        <div class="brand-info">
            <h1>Rihla</h1>
            <p>Morocco Travel Assistant</p>
        </div>
    </div>
    <div class="ai-badge">✨ AI-Powered</div>
</div>
""", unsafe_allow_html=True)

# Welcome Section
if len(st.session_state.messages) == 0 and not st.session_state.show_itinerary_form:
    st.markdown("""
    <div class="welcome-section">
        <h2 class="welcome-title">Welcome to Rihla! 👋</h2>
        <p class="welcome-subtitle">
            Your intelligent companion for exploring Morocco with personalized recommendations and custom itineraries.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input(
    "Ask about Morocco destinations, costs, or create an itinerary... 🇲🇦"
)

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.rerun()

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# BACKEND COMMUNICATION

if len(st.session_state.messages) > 0:
    last_message = st.session_state.messages[-1]

    if last_message["role"] == "user":
        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"text": last_message["content"]},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                ai_reply = data.get("response", "No response")

                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_reply}
                )
                st.rerun()
            else:
                st.error("Backend error")

        except Exception as e:
            st.error(f"Connection error: {e}")

# Sidebar
with st.sidebar:

    #  ITINERARY GENERATOR 
   
    st.markdown("### 🗓️ Create Itinerary")

    days = st.number_input(
        "Number of days",
        min_value=1,
        max_value=14,
        value=5
    )

    budget = st.selectbox(
        "Budget",
        ["budget", "moderate", "luxury"]
    )

    if st.button("Generate Itinerary ✨", use_container_width=True):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/itinerary",
                json={
                    "days": days,
                    "budget": budget
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                itinerary = data.get("itinerary")

                itinerary_text = f"🗺️ **{itinerary['title']}**\n\n"
                itinerary_text += f"**Budget:** {budget}\n"
                itinerary_text += f"**Cities:** {', '.join(itinerary['cities'])}\n\n"
                itinerary_text += "**Daily Plan:**\n"

                for day in itinerary["daily_plan"]:
                    itinerary_text += f"- **Day {day['day']} ({day['city']})**: "
                    itinerary_text += ", ".join(day["activities"]) + "\n"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": itinerary_text
                })
                st.rerun()
            else:
                st.error("Itinerary backend error")

        except Exception as e:
            st.error(f"Connection error: {e}")

    st.divider()

    # Destinations
    st.markdown("### 🗺️ Destinations")

    cities = [
        ("🏛️ Marrakech", "What should I visit in Marrakech?"),
        ("🕌 Fes", "Tell me about Fes"),
        ("💙 Chefchaouen", "What to see in Chefchaouen?"),
        ("🌊 Casablanca", "Tell me about Casablanca"),
        ("🏖️ Essaouira", "What to visit in Essaouira?"),
        ("🏜️ Sahara", "Tell me about the Sahara desert"),
        ("🎬 Ouarzazate", "What to see in Ouarzazate?"),
        ("⛰️ Atlas Mountains", "Tell me about Atlas Mountains"),
    ]

    for city, query in cities:
        if st.button(city, use_container_width=True):
            st.session_state.pending_query = query
            st.rerun()

    st.divider()
    st.caption("**Rihla v2.0**")
    st.caption("40 Destinations • Itinerary Generator")
