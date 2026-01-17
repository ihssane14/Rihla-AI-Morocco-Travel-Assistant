
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from huggingface_hub import InferenceClient
import os
import random
from dotenv import load_dotenv

load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rihla-morocco")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI(
    title="Rihla - رحلة",
    description="AI Travel Assistant for Morocco with HuggingFace LLM",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
print(" Loading models...")
try:
    # Embedding model for search
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(" Embedding model loaded")
    
    # Pinecone vector database
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print(" Pinecone connected")
    
    # HuggingFace LLM for conversation
    if HUGGINGFACE_API_KEY:
        hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)
        print(" HuggingFace LLM ready (Mistral-7B)")
    else:
        hf_client = None
        print(" HuggingFace API key not found - using fallback")
    
    print(" All services ready!")
except Exception as e:
    print(f" Warning: {e}")
    embedding_model = None
    index = None
    hf_client = None

# Pydantic Models
class Message(BaseModel):
    text: str = Field(..., min_length=1)

class ItineraryRequest(BaseModel):
    days: int = Field(..., ge=1, le=30)
    cities: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    budget: Optional[str] = "moderate"

class ChatResponse(BaseModel):
    success: bool
    response: str
    sources: Optional[List[dict]] = None
    error: Optional[str] = None

class ItineraryResponse(BaseModel):
    success: bool
    itinerary: Optional[dict] = None
    error: Optional[str] = None

# Itinerary templates
ITINERARY_TEMPLATES = {
    3: {
        "Marrakech": ["Jemaa el-Fnaa", "Jardin Majorelle", "Bahia Palace", "Marrakech Souks", "Koutoubia Mosque"],
        "Fes": ["Fes el-Bali (Old Medina)", "Chouara Tannery", "Bou Inania Madrasa"],
        "Chefchaouen": ["Blue Pearl Medina", "Spanish Mosque", "Ras El Maa"],
    },
    5: {
        "Marrakech": ["Jemaa el-Fnaa", "Jardin Majorelle", "Bahia Palace", "Marrakech Souks", "Saadian Tombs"],
        "Fes": ["Fes el-Bali (Old Medina)", "Chouara Tannery", "Bou Inania Madrasa", "Nejjarine Museum"],
        "Chefchaouen": ["Blue Pearl Medina", "Spanish Mosque", "Ras El Maa"],
        "Essaouira": ["Essaouira Medina", "Essaouira Port", "Skala de la Ville"],
    },
    7: {
        "Marrakech": ["Jemaa el-Fnaa", "Jardin Majorelle", "Bahia Palace", "Marrakech Souks"],
        "Sahara Desert": ["Erg Chebbi Dunes", "Desert Camp", "Khamlia Village"],
        "Fes": ["Fes el-Bali (Old Medina)", "Chouara Tannery", "Bou Inania Madrasa"],
        "Chefchaouen": ["Blue Pearl Medina", "Spanish Mosque"],
    },
    10: {
        "Marrakech": ["Jemaa el-Fnaa", "Jardin Majorelle", "Bahia Palace", "Marrakech Souks", "Saadian Tombs"],
        "Atlas Mountains": ["Imlil Village", "Ouzoud Waterfalls"],
        "Sahara Desert": ["Erg Chebbi Dunes", "Desert Camp", "Khamlia Village"],
        "Fes": ["Fes el-Bali (Old Medina)", "Chouara Tannery", "Bou Inania Madrasa"],
        "Chefchaouen": ["Blue Pearl Medina", "Spanish Mosque"],
        "Essaouira": ["Essaouira Medina", "Essaouira Port"],
    },
    14: {
        "Marrakech": ["Jemaa el-Fnaa", "Jardin Majorelle", "Bahia Palace", "Marrakech Souks", "Menara Gardens"],
        "Atlas Mountains": ["Imlil Village", "Ouzoud Waterfalls"],
        "Sahara Desert": ["Erg Chebbi Dunes", "Desert Camp", "Khamlia Village"],
        "Ouarzazate": ["Ait Ben Haddou", "Atlas Film Studios"],
        "Fes": ["Fes el-Bali (Old Medina)", "Chouara Tannery", "Bou Inania Madrasa", "Nejjarine Museum"],
        "Meknes": ["Bab Mansour", "Volubilis"],
        "Chefchaouen": ["Blue Pearl Medina", "Spanish Mosque", "Ras El Maa"],
        "Tangier": ["Cape Spartel", "Tangier Medina"],
        "Essaouira": ["Essaouira Medina", "Essaouira Port", "Skala de la Ville"],
    }
}

def generate_llm_response(query: str, context_data: List[dict]) -> str:
    """
    Generate natural conversational response using HuggingFace LLM
    
    Args:
        query: User's question
        context_data: List of relevant destinations from Pinecone
    
    Returns:
        Natural language response from LLM
    """
    
    if not hf_client:
        # Fallback to formatted response if LLM not available
        return generate_fallback_response(query, context_data)
    
    try:
        # Prepare context from Pinecone results
        context_text = ""
        for i, dest in enumerate(context_data, 1):
            context_text += f"""
Destination {i}:
- Place: {dest.get('place', 'Unknown')}
- City: {dest.get('city', 'Unknown')}
- Type: {dest.get('type', 'General')}
- Description: {dest.get('description', '')}
- Tips: {dest.get('tips', '')}
- Cost: {dest.get('cost', '')}
- Best Time: {dest.get('best_time', 'Anytime')}

"""
        
        # Create prompt for LLM
        prompt = f"""<s>[INST] You are Rihla, a friendly and knowledgeable Morocco travel assistant. 

User asks: "{query}"

Here is relevant information about Morocco destinations:
{context_text}

Provide a helpful, conversational response about Morocco travel. Be friendly, informative, and concise (max 300 words). Include practical tips and recommendations based on the context provided.

If the user asks about places not in the context, mention that you have information about 40+ destinations in Morocco and suggest they ask about specific cities.

Response: [/INST]"""
        
        # Generate response from LLM
        response = hf_client.text_generation(
            prompt,
            model=LLM_MODEL,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        
        # Clean up response
        response = response.strip()
        
        # Add emoji for better UX
        response = f"🇲🇦 {response}"
        
        return response
        
    except Exception as e:
        print(f" LLM Error: {e}")
        # Fallback to formatted response
        return generate_fallback_response(query, context_data)

def generate_fallback_response(query: str, context_data: List[dict]) -> str:
    """Fallback formatted response if LLM fails"""
    
    if not context_data:
        return """I understand you're asking about Morocco travel, but I don't have specific information about that in my database right now.

I have detailed information about 40+ destinations across Morocco including:
- Imperial Cities: Marrakech, Fes, Meknes, Rabat
- Coastal Cities: Essaouira, Tangier, Chefchaouen
- Desert & Mountains: Sahara, Atlas Mountains, Ouarzazate

Try asking about a specific city or attraction!"""
    
    response_parts = []
    response_parts.append("Here's what I found about Morocco destinations:\n")
    
    for i, dest in enumerate(context_data, 1):
        place = dest.get('place', 'Unknown')
        city = dest.get('city', 'Unknown')
        desc = dest.get('description', '')
        tips = dest.get('tips', '')
        cost = dest.get('cost', '')
        
        response_parts.append(f"\n**{i}. {place}** ({city})")
        response_parts.append(f"{desc}")
        response_parts.append(f"💡 Tips: {tips}")
        response_parts.append(f"💰 Cost: {cost}\n")
    
    return "\n".join(response_parts)

# Routes
@app.get("/")
def root():
    """API root endpoint"""
    return {
        "message": "Welcome to Rihla! 🇲🇦 مرحباً بك في رحلة",
        "version": "3.0.0 - With HuggingFace LLM",
        "status": "active",
        "ai_model": LLM_MODEL if hf_client else "Fallback Mode",
        "destinations": 40,
        "endpoints": {
            "chat": "/chat",
            "itinerary": "/itinerary",
            "health": "/health",
            "cities": "/cities",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_model": embedding_model is not None,
        "pinecone": index is not None,
        "llm": hf_client is not None,
        "destinations": 40,
        "version": "3.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    """
    Main chat endpoint with AI-powered responses
    Now uses HuggingFace LLM for natural conversations!
    """
    
    if not embedding_model or not index:
        raise HTTPException(503, "Services not available")
    
    try:
        print(f"\n Question: {message.text}")
        
        # Check for itinerary request
        query_lower = message.text.lower()
        itinerary_keywords = ['itinerary', 'plan', 'trip', 'days', 'week', 'schedule', 'برنامج', 'خطة']
        
        if any(keyword in query_lower for keyword in itinerary_keywords):
            return ChatResponse(
                success=True,
                response="""🗓️ I can help you plan your perfect Morocco trip!

**Available Itineraries:**
• 3-5 days: Marrakech highlights
• 7 days: Classic tour with Sahara
• 10 days: Comprehensive experience
• 14 days: Complete Morocco journey

Just tell me how many days you have, and I'll create a custom itinerary for you!

Example: "Create a 7-day itinerary" """,
                sources=None
            )
        
        # Search Pinecone for relevant destinations
        query_embedding = embedding_model.encode(message.text).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        RELEVANCE_THRESHOLD = 0.35
        relevant_matches = [m for m in results.matches if m.score >= RELEVANCE_THRESHOLD]
        
        # Detect mentioned city
        cities = [
            'marrakech', 'fes', 'fez', 'chefchaouen', 'chaouen',
            'casablanca', 'casa', 'essaouira', 'sahara', 'merzouga',
            'rabat', 'tangier', 'tanger', 'agadir', 'meknes', 'tetouan',
            'ouarzazate', 'atlas', 'dades', 'todra', 'asilah', 'imlil'
        ]
        
        mentioned_city = None
        for city in cities:
            if city in query_lower:
                mentioned_city = city
                break
        
        # Filter by city if mentioned
        if relevant_matches and mentioned_city:
            city_matches = [
                m for m in relevant_matches 
                if mentioned_city in m.metadata.get('city', '').lower() 
                or mentioned_city in m.metadata.get('place', '').lower()
            ]
            if city_matches:
                relevant_matches = city_matches[:3]
            else:
                relevant_matches = relevant_matches[:3]
        elif relevant_matches:
            relevant_matches = relevant_matches[:3]
        
        # Prepare sources for response
        sources = []
        context_data = []
        
        for match in relevant_matches:
            m = match.metadata
            context_data.append(m)
            sources.append({
                "place": m.get('place', 'Unknown'),
                "city": m.get('city', 'Unknown'),
                "relevance_score": round(match.score, 2)
            })
        
        # Generate AI response using LLM
        print("🤖 Generating AI response...")
        response_text = generate_llm_response(message.text, context_data)
        print(" Response generated")
        
        return ChatResponse(
            success=True,
            response=response_text,
            sources=sources if sources else None
        )
        
    except Exception as e:
        print(f" Error: {e}")
        return ChatResponse(
            success=False,
            response="I apologize! I encountered a technical issue. Could you try rephrasing your question?",
            error=str(e)
        )

@app.post("/itinerary", response_model=ItineraryResponse)
async def generate_itinerary(request: ItineraryRequest):
    """Generate custom Morocco itinerary"""
    
    try:
        days = request.days
        
        # Select template based on days
        if days <= 3:
            template = ITINERARY_TEMPLATES[3]
        elif days <= 5:
            template = ITINERARY_TEMPLATES[5]
        elif days <= 7:
            template = ITINERARY_TEMPLATES[7]
        elif days <= 10:
            template = ITINERARY_TEMPLATES[10]
        else:
            template = ITINERARY_TEMPLATES[14]
        
        # Build itinerary
        itinerary = {
            "title": f"{days}-Day Morocco Itinerary",
            "total_days": days,
            "cities": list(template.keys()),
            "daily_plan": [],
            "estimated_cost": {
                "budget": f"{days * 300}-{days * 500} MAD/day",
                "moderate": f"{days * 600}-{days * 900} MAD/day",
                "luxury": f"{days * 1200}+ MAD/day"
            },
            "tips": [
                "Book accommodations in advance, especially in Marrakech and Fes",
                "Hire official guides in medinas to avoid getting lost",
                "Bargain in souks - start at 50% of asking price",
                "Bring layers - desert nights are cold",
                "Try local specialties: tagine, couscous, mint tea"
            ]
        }
        
        day_counter = 1
        cities_list = list(template.items())
        days_per_city = max(1, days // len(cities_list))
        
        for city, places in cities_list:
            if day_counter > days:
                break
            
            city_days = min(days_per_city, days - day_counter + 1)
            
            for day_in_city in range(city_days):
                if day_counter > days:
                    break
                
                day_places = places[:3] if len(places) > 3 else places
                
                itinerary["daily_plan"].append({
                    "day": day_counter,
                    "city": city,
                    "title": f"Day {day_counter}: {city}",
                    "activities": day_places,
                    "description": f"Explore the highlights of {city}",
                    "meals": "Try local restaurants in medina",
                    "accommodation": f"Stay in {city} (riad/hotel)"
                })
                
                day_counter += 1
        
        return ItineraryResponse(
            success=True,
            itinerary=itinerary
        )
        
    except Exception as e:
        return ItineraryResponse(
            success=False,
            error=str(e)
        )

@app.get("/cities")
def get_cities():
    """Get available cities"""
    return {
        "success": True,
        "total": 15,
        "cities": [
            {"name": "Marrakech", "emoji": "🏛️", "places": 8},
            {"name": "Fes", "emoji": "🕌", "places": 4},
            {"name": "Chefchaouen", "emoji": "💙", "places": 3},
            {"name": "Casablanca", "emoji": "🌊", "places": 2},
            {"name": "Essaouira", "emoji": "🏖️", "places": 3},
            {"name": "Sahara Desert", "emoji": "🏜️", "places": 3},
            {"name": "Ouarzazate", "emoji": "🎬", "places": 2},
            {"name": "Agadir", "emoji": "🏖️", "places": 2},
            {"name": "Tangier", "emoji": "🌅", "places": 2},
            {"name": "Meknes", "emoji": "🏰", "places": 2},
            {"name": "Rabat", "emoji": "🏛️", "places": 3},
            {"name": "Atlas Mountains", "emoji": "⛰️", "places": 2},
            {"name": "Dades Valley", "emoji": "🏞️", "places": 1},
            {"name": "Todra Gorge", "emoji": "🏞️", "places": 1},
            {"name": "Asilah", "emoji": "🎨", "places": 1}
        ]
    }

@app.get("/stats")
def get_stats():
    """Database statistics"""
    try:
        stats = index.describe_index_stats()
        return {
            "success": True,
            "total_destinations": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_name": PINECONE_INDEX_NAME,
            "model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL if hf_client else "Not configured",
            "features": ["Chat with LLM", "Itinerary Generator", "40 Destinations", "Vector Search"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
   
    print(" Starting Rihla V3 - Morocco Travel Assistant")
    print("="*60)
    print("AI Model: " + (LLM_MODEL if hf_client else "Fallback Mode"))
    print(" API Docs: http://127.0.0.1:8000/docs")
    print(" Homepage: http://127.0.0.1:8000")
    print(" Chat: http://127.0.0.1:8000/chat")
    print(" Itinerary: http://127.0.0.1:8000/itinerary")
    print(" 40 Destinations Ready")
    uvicorn.run(app, host="0.0.0.0", port=8000)