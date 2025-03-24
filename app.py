import os
from typing import List, Dict, Any, Tuple, Optional
import gradio as gr

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

import langgraph.graph as lg
from langgraph.graph import END, StateGraph

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
        progress_updates: List[str] = None,
    ):
        self.query = query
        self.search_results = search_results or []
        self.research_notes = research_notes or []
        self.historical_persona = historical_persona or {}
        self.narrative_style = narrative_style
        self.current_documents = current_documents or []
        self.conversation_history = conversation_history or []
        self.final_response = final_response
        self.progress_updates = progress_updates or []
        super().__init__(
            query=self.query,
            search_results=self.search_results,
            research_notes=self.research_notes,
            historical_persona=self.historical_persona,
            narrative_style=self.narrative_style,
            current_documents=self.current_documents,
            conversation_history=self.conversation_history,
            final_response=self.final_response,
            progress_updates=self.progress_updates,
        )

# Node functions
def determine_research_topic(state: AgentState) -> AgentState:
    """Extract the research topic from the user query."""
    state["progress_updates"].append("Determining research topic...")
    
    prompt = ChatPromptTemplate.from_template(
        "Extract the main research topic from the following query: {query}\n\n"
        "Return only the core research topic or subject."
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) | StrOutputParser()
    topic = chain.invoke({"query": state["query"]})
    
    # Record in conversation history
    state["conversation_history"].append({"role": "system", "content": f"Research topic identified: {topic}"})
    state["progress_updates"].append(f"Research topic identified: {topic}")
    
    return state

def select_historical_persona(state: AgentState) -> AgentState:
    """Select an appropriate historical persona based on the research topic."""
    state["progress_updates"].append("Selecting historical persona...")
    
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic: {query}\n\n"
        "Select the most appropriate historical era from the following options:\n"
        "- ancient_rome: Ancient Rome (philosopher's perspective)\n"
        "- renaissance: Renaissance Italy (artist/scientist perspective)\n"
        "- victorian: Victorian England (mathematician's perspective)\n"
        "- modern_science: 20th Century America (physicist's perspective)\n\n"
        "Return only one of these exact options based on which would be most appropriate and interesting."
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2) | StrOutputParser()
    selected_era = chain.invoke({"query": state["query"]})
    
    # Fallback if the model doesn't return a valid option
    if selected_era not in HISTORICAL_PERSONAS:
        selected_era = "modern_science"
    
    state["historical_persona"] = HISTORICAL_PERSONAS[selected_era]
    
    # Record in conversation history
    update_msg = f"Selected historical persona: {state['historical_persona']['name']} from {state['historical_persona']['era']}"
    state["conversation_history"].append({"role": "system", "content": update_msg})
    state["progress_updates"].append(update_msg)
    
    return state

def search_for_information(state: AgentState) -> AgentState:
    """Perform a search on the research topic."""
    state["progress_updates"].append("Searching for information...")
    
    # Direct API key usage for Tavily
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # Provide an informative message if API key is missing
        state["progress_updates"].append("Error: Tavily API key not found. Using mock search results instead.")
        state["search_results"] = [
            {"url": "https://en.wikipedia.org/wiki/Main_Page", "content": "Mock content for testing purposes"},
            {"url": "https://www.britannica.com/", "content": "Another mock result for testing"}
        ]
    else:
        try:
            search_tool = TavilySearchResults(tavily_api_key=api_key, max_results=5)
            search_results = search_tool.invoke(state["query"])
            state["search_results"] = search_results
            state["progress_updates"].append(f"Found {len(search_results)} relevant sources of information")
        except Exception as e:
            state["progress_updates"].append(f"Search error: {str(e)}. Using mock results instead.")
            state["search_results"] = [
                {"url": "https://en.wikipedia.org/wiki/Main_Page", "content": "Mock content for testing purposes"},
                {"url": "https://www.britannica.com/", "content": "Another mock result for testing"}
            ]
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Found {len(state['search_results'])} relevant sources of information"
    })
    
    return state

def extract_relevant_content(state: AgentState) -> AgentState:
    """Extract and download relevant content from search results."""
    state["progress_updates"].append("Extracting content from sources...")
    
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
            state["progress_updates"].append(f"Extracted content from {url}")
        except Exception as e:
            state["progress_updates"].append(f"Error loading URL {url}: {str(e)}")
    
    state["current_documents"] = all_documents
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Extracted content from {len(all_documents)} document chunks"
    })
    
    return state

def synthesize_research_notes(state: AgentState) -> AgentState:
    """Synthesize research notes from the documents."""
    state["progress_updates"].append("Synthesizing research notes...")
    
    if not state["current_documents"]:
        state["research_notes"] = ["No documents were successfully retrieved. Using general knowledge instead."]
        state["progress_updates"].append("No documents retrieved - using general knowledge")
        
        # Fall back to general knowledge
        prompt = ChatPromptTemplate.from_template(
            "You are a research assistant helping to gather information on: {query}\n\n"
            "Please create a comprehensive set of research notes about this topic using your general knowledge. "
            "Include important facts, dates, figures, and concepts. Organize the information in a clear, logical way."
        )
        
        chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash") | StrOutputParser()
        research_notes = chain.invoke({"query": state["query"]})
        
        state["research_notes"] = [research_notes]
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
    
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2) | StrOutputParser()
    research_notes = chain.invoke({"query": state["query"], "documents": combined_text})
    
    state["research_notes"] = [research_notes]
    state["progress_updates"].append("Research notes synthesized from collected documents")
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": "Research notes synthesized from collected documents"
    })
    
    return state

def select_narrative_style(state: AgentState) -> AgentState:
    """Select an appropriate narrative style for the final response."""
    state["progress_updates"].append("Selecting narrative style...")
    
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic: {query}\n\n"
        "Select the most engaging narrative style from the following options:\n"
        "- adventure: Present the information as an exciting adventure with the user as a protagonist\n"
        "- dialogue: Present the information as a dialogue between the historical persona and the user\n"
        "- journal: Present the information as journal entries from the historical persona\n"
        "- letter: Present the information as a letter from the historical persona to the user\n\n"
        "Return only one of these exact options based on which would be most appropriate and interesting."
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3) | StrOutputParser()
    selected_style = chain.invoke({"query": state["query"]})
    
    # Fallback if the model doesn't return a valid option
    valid_styles = ["adventure", "dialogue", "journal", "letter"]
    if selected_style not in valid_styles:
        selected_style = "dialogue"
    
    state["narrative_style"] = selected_style
    state["progress_updates"].append(f"Selected narrative style: {state['narrative_style']}")
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "system", 
        "content": f"Selected narrative style: {state['narrative_style']}"
    })
    
    return state

def generate_time_travel_response(state: AgentState) -> AgentState:
    """Generate the final time-traveling research response."""
    state["progress_updates"].append("Generating time-travel response...")
    
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
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7) | StrOutputParser()
    
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
    state["progress_updates"].append("Time-travel research response generated")
    
    # Record in conversation history
    state["conversation_history"].append({
        "role": "assistant", 
        "content": "Time-travel research response generated"
    })
    
    return state

def should_generate_visuals(state: AgentState) -> str:
    """Determine if we should generate visual aids."""
    state["progress_updates"].append("Determining if visuals would enhance the response...")
    
    # Check if query suggests visual content would be helpful
    prompt = ChatPromptTemplate.from_template(
        "Based on the research topic: {query}\n\n"
        "Would visual aids like timelines, maps, or diagrams significantly enhance understanding of this topic? "
        "Answer with just 'yes' or 'no'."
    )
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) | StrOutputParser()
    response = chain.invoke({"query": state["query"]})
    
    if "yes" in response.lower():
        state["progress_updates"].append("Decision: Visual aids would be helpful")
        return "generate_visuals"
    else:
        state["progress_updates"].append("Decision: Visual aids not necessary")
        return "finalize_response"

def generate_visual_aids(state: AgentState) -> AgentState:
    """Generate text-based visual aids like ASCII art, markdown diagrams, etc."""
    state["progress_updates"].append("Generating visual aids...")
    
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
    
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4) | StrOutputParser()
    visual_aid = chain.invoke({
        "query": state["query"],
        "research_notes": "\n\n".join(state["research_notes"])
    })
    
    # Add the visual aid to the final response
    state["final_response"] += "\n\n" + visual_aid
    state["progress_updates"].append("Visual aid generated and added to response")
    
    return state

def finalize_response(state: AgentState) -> AgentState:
    """Add final touches to the response."""
    state["progress_updates"].append("Finalizing response...")
    
    prompt = ChatPromptTemplate.from_template(
        "Here is a time-traveling research response:\n\n{response}\n\n"
        "Add a brief closing note as {persona_name} that encourages the user to continue exploring this topic "
        "and mentions how this knowledge connects to modern understanding. "
        "Keep it in character as {persona_name} from {persona_era}."
    )
    
    chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5) | StrOutputParser()
    closing = chain.invoke({
        "response": state["final_response"],
        "persona_name": state["historical_persona"]["name"],
        "persona_era": state["historical_persona"]["era"]
    })
    
    state["final_response"] += "\n\n" + closing
    state["progress_updates"].append("Response finalized")
    
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

# Function to run the agent with updating progress
def run_time_travel_research_agent(query: str, progress_callback=None):
    """
    Run the Time-Traveling Research Companion Agent with a user query.
    
    Args:
        query: The research query
        progress_callback: Optional callback function to report progress updates
    
    Returns:
        The final response and progress updates
    """
    graph = build_research_companion_graph()
    
    # Initialize state with query
    initial_state = AgentState(query=query)
    
    # Execute the graph
    result = graph.invoke(initial_state)
    
    # If there's a callback function, send final updates
    if progress_callback:
        progress_callback(result["progress_updates"])
    
    return result["final_response"], result["progress_updates"]

# Gradio interface
def create_gradio_interface():
    """Create a Gradio interface for the Time-Traveling Research Companion."""
    
    # Initialize progress updates
    progress_updates = ["Initializing..."]
    
    def research_query(query, api_key=None):
        nonlocal progress_updates
        progress_updates = ["Starting research process..."]
        
        # Set API key if provided
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key
        
        # Removed environment variable dependency assumption
        def progress_callback(updates):
            nonlocal progress_updates
            progress_updates = updates
        
        try:
            # Add fallback for missing API keys
            if not os.getenv("GOOGLE_API_KEY"):
                return (
                    "ERROR: Google API key not found. Please set your GOOGLE_API_KEY environment variable.", 
                    "Google API key not found."
                )
            
            response, updates = run_time_travel_research_agent(query, progress_callback)
            return response, "\n".join(progress_updates)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            progress_updates.append(error_message)
            return (
                f"I encountered an error while processing your request:\n\n{error_message}", 
                "\n".join(progress_updates)
            )
        
    with gr.Blocks(title="Time-Traveling Research Companion") as demo:
        gr.Markdown("# Time-Traveling Research Companion")
        gr.Markdown("""
        This AI agent uses LangChain and LangGraph to create an immersive research experience. 
        Enter a research topic, and a historical persona will guide you through the information in a creative way!
        """)
        
        # In the create_gradio_interface function
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="What would you like to research?",
                    placeholder="Enter a topic like 'ancient Egyptian architecture' or 'quantum computing'...",
                    lines=2
                )
                api_key_input = gr.Textbox(
                    label="Tavily API Key (optional)",
                    placeholder="Enter your Tavily API key if you have one",
                    type="password"
                )
                submit_btn = gr.Button("Begin Time Travel", variant="primary")
            
            with gr.Column(scale=3):
                output = gr.Markdown(label="Time-Traveling Research Results")

        with gr.Accordion("Progress Updates", open=False):
            progress_output = gr.Markdown()

        submit_btn.click(
            fn=research_query,
            inputs=[query_input, api_key_input],
            outputs=[output, progress_output]
        )
        
        submit_btn.click(
            fn=research_query,
            inputs=[query_input, api_key_input],
            outputs=[output, progress_output]
        )
    
    gr.Markdown("""
    ### How it works
    
    1. The agent identifies your research topic
    2. It selects an appropriate historical persona (Marcus Aurelius, Leonardo da Vinci, Ada Lovelace, or Richard Feynman)
    3. It searches for information (with Tavily if API key provided)
    4. It synthesizes research notes and selects a narrative style
    5. It presents the information through the eyes of your historical guide
    
    ### Notes
    - Providing a Tavily API key will enable real-time web search for your topic
    - Without a Tavily key, the agent will use its general knowledge
    - Google API key must be set as an environment variable
    """)
    return demo

# Main function
def main():
    """Main function to run the Gradio interface."""
    demo = create_gradio_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()