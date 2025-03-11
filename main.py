import os
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import init_chat_model


import langgraph.graph as lg
from langgraph.graph import END, StateGraph

# Configuration
# API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3I9V84iorKbFPeOqk_SP2o_ojTc5eRn0"
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = "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"
# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
search_tool = TavilySearchResults(max_results=5)

# Historical personas
HISTORICAL_PERSONAS = {
    "ancient_rome": {
        "name": "Marcus Aurelius",
        "era": "Ancient Rome",
        "tone": "philosophical and stoic",
        "perspective": "I view the world through the lens of virtue and duty. As a Stoic emperor, I focus on what is within our control and accept what is not."
    },
    "renaissance": {
        "name": "Leonardo da Vinci",
        "era": "Renaissance Italy",
        "tone": "curious and analytical",
        "perspective": "I see connections between art and science, and believe careful observation is the key to understanding nature's secrets."
    },
    "victorian": {
        "name": "Ada Lovelace",
        "era": "Victorian England",
        "tone": "methodical and forward-thinking",
        "perspective": "I see mathematics as poetical science and believe machines could one day do more than calculations - perhaps even create art."
    },
    "modern_science": {
        "name": "Richard Feynman",
        "era": "20th Century America",
        "tone": "excited and playful",
        "perspective": "I believe if you can't explain something simply, you don't understand it well enough. Science should be fun!"
    }
}

# State definition
class AgentState(dict):
    """State for the Research Agent."""
    def __init__(
        self,
        query: str = "",
        search_results: List[Dict] = None,
        research_notes: List[str] = None,
        historical_persona: Dict = None,
        narrative_style: str = "default",
        current_documents: List[Document] = None,
        conversation_history: List[Dict] = None,
        final_response: str = "",
    ):
        self.query = query
        self.search_results = search_results or []
        self.research_notes = research_notes or []
        self.historical_persona = historical_persona or {}
        self.narrative_style = narrative_style
        self.current_documents = current_documents or []
        self.conversation_history = conversation_history or []
        self.final_response = final_response
        super().__init__(
            query=self.query,
            search_results=self.search_results,
            research_notes=self.research_notes,
            historical_persona=self.historical_persona,
            narrative_style=self.narrative_style,
            current_documents=self.current_documents,
            conversation_history=self.conversation_history,
            final_response=self.final_response,
        )

# Node functions
def determine_research_topic(state: AgentState) -> AgentState:
    """Extract the research topic from the user query."""
    prompt = ChatPromptTemplate.from_template(
        "Extract the main research topic from the following query: {query}\n\n"
        "Return only the core research topic or subject."
    )
    chain = prompt | llm | StrOutputParser()
    topic = chain.invoke({"query": state["query"]})
    
    # Record in conversation history
    state["conversation_history"].append({"role": "system", "content": f"Research topic identified: {topic}"})
    
    return state

def select_historical_persona(state: AgentState) -> AgentState:
    """Select an appropriate historical persona based on the research topic."""
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic: {query}\n\n"
        "Select the most appropriate historical era from the following options:\n"
        "- ancient_rome: Ancient Rome (philosopher's perspective)\n"
        "- renaissance: Renaissance Italy (artist/scientist perspective)\n"
        "- victorian: Victorian England (mathematician's perspective)\n"
        "- modern_science: 20th Century America (physicist's perspective)\n\n"
        "Return only one of these exact options based on which would be most appropriate and interesting."
    )
    chain = prompt | llm | StrOutputParser()
    selected_era = chain.invoke({"query": state["query"]})
    
    # Fallback if the model doesn't return a valid option
    if selected_era not in HISTORICAL_PERSONAS:
        selected_era = "modern_science"
    
    state["historical_persona"] = HISTORICAL_PERSONAS[selected_era]
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Selected historical persona: {state['historical_persona']['name']} from {state['historical_persona']['era']}"
    })
    
    return state

def search_for_information(state: AgentState) -> AgentState:
    """Perform a search on the research topic."""
    search_results = search_tool.invoke(state["query"])
    state["search_results"] = search_results
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Found {len(search_results)} relevant sources of information"
    })
    
    return state

def extract_relevant_content(state: AgentState) -> AgentState:
    """Extract and download relevant content from search results."""
    urls = [result["url"] for result in state["search_results"]]
    all_documents = []
    
    for url in urls[:3]:  # Limit to first 3 to avoid overloading
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Split the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)
            
            all_documents.extend(split_docs)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")
    
    state["current_documents"] = all_documents
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Extracted content from {len(all_documents)} document chunks"
    })
    
    return state

def synthesize_research_notes(state: AgentState) -> AgentState:
    """Synthesize research notes from the documents."""
    if not state["current_documents"]:
        state["research_notes"] = ["No documents were successfully retrieved. Please try a different search query."]
        return state
    
    # Combine document texts with their metadata
    combined_text = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
        for doc in state["current_documents"][:5]  # Limit processing to first 5 chunks
    ])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant helping to gather information on: {query}\n\n"
        "Here are some documents related to this topic:\n{documents}\n\n"
        "Please create a comprehensive set of research notes that summarizes the key information from these documents. "
        "Include important facts, dates, figures, and concepts. Organize the information in a clear, logical way."
    )
    
    chain = prompt | llm | StrOutputParser()
    research_notes = chain.invoke({"query": state["query"], "documents": combined_text})
    
    state["research_notes"] = [research_notes]
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": "Research notes synthesized from collected documents"
    })
    
    return state

def select_narrative_style(state: AgentState) -> AgentState:
    """Select an appropriate narrative style for the final response."""
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic: {query}\n\n"
        "Select the most engaging narrative style from the following options:\n"
        "- adventure: Present the information as an exciting adventure with the user as a protagonist\n"
        "- dialogue: Present the information as a dialogue between the historical persona and the user\n"
        "- journal: Present the information as journal entries from the historical persona\n"
        "- letter: Present the information as a letter from the historical persona to the user\n\n"
        "Return only one of these exact options based on which would be most appropriate and interesting."
    )
    chain = prompt | llm | StrOutputParser()
    selected_style = chain.invoke({"query": state["query"]})
    
    # Fallback if the model doesn't return a valid option
    valid_styles = ["adventure", "dialogue", "journal", "letter"]
    if selected_style not in valid_styles:
        selected_style = "dialogue"
    
    state["narrative_style"] = selected_style
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Selected narrative style: {state['narrative_style']}"
    })
    
    return state

def generate_time_travel_response(state: AgentState) -> AgentState:
    """Generate the final time-traveling research response."""
    persona = state["historical_persona"]
    style = state["narrative_style"]
    notes = "\n\n".join(state["research_notes"])
    
    prompt_template = """
    You are a time-traveling research companion helping a user learn about: {query}
    
    You will take on the persona of {persona_name} from {persona_era}, and present the research findings in an engaging way.
    
    PERSONA DETAILS:
    - Name: {persona_name}
    - Historical era: {persona_era}
    - Tone: {persona_tone}
    - Perspective: {persona_perspective}
    
    PRESENTATION STYLE: {narrative_style}
    
    RESEARCH NOTES:
    {research_notes}
    
    Now, create an engaging, educational response that:
    1. Introduces yourself as the historical persona
    2. Presents the research information in the selected narrative style
    3. Includes factual information from the research notes
    4. Uses language, metaphors, and references appropriate to your historical persona
    5. Makes the learning experience fun and memorable
    6. Ends with 2-3 thought-provoking questions that encourage deeper exploration of the topic
    
    Be creative, informative, and historically authentic while ensuring accuracy of the factual content.
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    final_response = chain.invoke({
        "query": state["query"],
        "persona_name": persona["name"],
        "persona_era": persona["era"],
        "persona_tone": persona["tone"],
        "persona_perspective": persona["perspective"],
        "narrative_style": style,
        "research_notes": notes
    })
    
    state["final_response"] = final_response
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "assistant", 
        "content": "Time-travel research response generated"
    })
    
    return state

def should_generate_visuals(state: AgentState) -> str:
    """Determine if we should generate visual aids."""
    # Check if query suggests visual content would be helpful
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic: {query}\n\n"
        "Would visual aids like timelines, maps, or diagrams significantly enhance understanding of this topic? "
        "Answer with just 'yes' or 'no'."
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": state["query"]})
    
    if "yes" in response.lower():
        return "generate_visuals"
    else:
        return "finalize_response"

def generate_visual_aids(state: AgentState) -> AgentState:
    """Generate text-based visual aids like ASCII art, markdown diagrams, etc."""
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic and notes:\n\n"
        "Topic: {query}\n"
        "Notes: {research_notes}\n\n"
        "Generate ONE visual representation that would help illustrate a key concept from this research. "
        "You can create:\n"
        "1. An ASCII art diagram\n"
        "2. A simple markdown table\n"
        "3. A text-based timeline\n"
        "4. A simple markdown chart\n\n"
        "Choose the most appropriate visual format for this topic and create one visual aid that "
        "enhances understanding of a key concept or relationship. Be creative but ensure the visual "
        "is clear and informative."
    )
    
    chain = prompt | llm | StrOutputParser()
    visual_aid = chain.invoke({
        "query": state["query"],
        "research_notes": "\n\n".join(state["research_notes"])
    })
    
    # Add the visual aid to the final response
    state["final_response"] += "\n\n" + visual_aid
    
    return state

def finalize_response(state: AgentState) -> AgentState:
    """Add final touches to the response."""
    prompt = ChatPromptTemplate.from_template(
        "Here is a time-traveling research response:\n\n{response}\n\n"
        "Add a brief closing note as {persona_name} that encourages the user to continue exploring this topic "
        "and mentions how this knowledge connects to modern understanding. "
        "Keep it in character as {persona_name} from {persona_era}."
    )
    
    chain = prompt | llm | StrOutputParser()
    closing = chain.invoke({
        "response": state["final_response"],
        "persona_name": state["historical_persona"]["name"],
        "persona_era": state["historical_persona"]["era"]
    })
    
    state["final_response"] += "\n\n" + closing
    
    return state

# Create the graph
def build_research_companion_graph():
    """Build the LangGraph for the Time-Traveling Research Companion."""
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("determine_research_topic", determine_research_topic)
    workflow.add_node("select_historical_persona", select_historical_persona)
    workflow.add_node("search_for_information", search_for_information)
    workflow.add_node("extract_relevant_content", extract_relevant_content)
    workflow.add_node("synthesize_research_notes", synthesize_research_notes)
    workflow.add_node("select_narrative_style", select_narrative_style)
    workflow.add_node("generate_time_travel_response", generate_time_travel_response)
    workflow.add_node("generate_visual_aids", generate_visual_aids)
    workflow.add_node("finalize_response", finalize_response)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "generate_time_travel_response",
        should_generate_visuals,
        {
            "generate_visuals": "generate_visual_aids",
            "finalize_response": "finalize_response"
        }
    )
    
    # Add regular edges
    workflow.add_edge("determine_research_topic", "select_historical_persona")
    workflow.add_edge("select_historical_persona", "search_for_information")
    workflow.add_edge("search_for_information", "extract_relevant_content")
    workflow.add_edge("extract_relevant_content", "synthesize_research_notes")
    workflow.add_edge("synthesize_research_notes", "select_narrative_style")
    workflow.add_edge("select_narrative_style", "generate_time_travel_response")
    workflow.add_edge("generate_visual_aids", "finalize_response")
    workflow.add_edge("finalize_response", END)
    
    # Set the entry point
    workflow.set_entry_point("determine_research_topic")
    
    return workflow.compile()

# Function to run the agent
def run_time_travel_research_agent(query: str) -> str:
    """Run the Time-Traveling Research Companion Agent with a user query."""
    graph = build_research_companion_graph()
    
    # Initialize state with query
    initial_state = AgentState(query=query)
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    return result["final_response"]

# Example usage function with proper handling of API keys
def main():
    """Main function to demonstrate the agent's capabilities."""
    print("Welcome to the Time-Traveling Research Companion!")
    print("Please ensure you have set your GOOGLE_API_KEY and TAVILY_API_KEY environment variables.\n")
    
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("API keys not found. Please set your environment variables.")
        return
    
    query = input("What would you like to research today? ")
    print("\nInitiating time travel sequence...")
    print("This may take a moment as we journey through history...\n")
    
    try:
        response = run_time_travel_research_agent(query)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your API keys and internet connection and try again.")

if __name__ == "__main__":
    main()