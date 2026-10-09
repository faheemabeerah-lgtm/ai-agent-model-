import os
import streamlit as st

# Fix the CrewAI/Groq cache_breakpoint error.
# IMPORTANT: run this before importing Agent, Crew, LLM, or Task.
import crewai.llms.cache as crew_cache

crew_cache.mark_cache_breakpoint = lambda msg: msg

from crewai import Agent, Crew, LLM, Process, Task
from ddgs import DDGS


# -----------------------------
# Streamlit page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI Research Agent")
st.write(
    "Enter a research topic to search the web "
    "and generate a structured research report."
)


# -----------------------------
# Configure Groq
# -----------------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. "
        "Add it in Streamlit Cloud → Manage app → Settings → Secrets."
    )
    st.stop()

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=api_key
)


# -----------------------------
# Research topic input
# -----------------------------
topic = st.text_area(
    "Enter your research topic",
    placeholder="Example: The impact of AI on education",
    height=100
)

generate = st.button(
    "Generate Research Report",
    type="primary"
)


# -----------------------------
# Search the web
# -----------------------------
def search_web(query):
    results = []

    with DDGS() as search:
        for item in search.text(query, max_results=8):
            results.append({
                "title": item.get("title", "Untitled"),
                "url": item.get("href", ""),
                "snippet": item.get("body", "")
            })

    return results


# -----------------------------
# Generate research report
# -----------------------------
if generate:
    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    try:
        with st.spinner("Searching the web..."):
            search_results = search_web(topic)

        if not search_results:
            st.warning(
                "No search results were found. "
                "Please try another topic."
            )
            st.stop()

        # Give the agent the actual search results.
        sources_text = "\n\n".join(
            f"Title: {item['title']}\n"
            f"URL: {item['url']}\n"
            f"Snippet: {item['snippet']}"
            for item in search_results
        )

        researcher = Agent(
            role="AI Research Analyst",
            goal=(
                "Create accurate, clear, well-organized research "
                "reports supported by the provided web sources."
            ),
            backstory=(
                "You are a careful research analyst. "
                "You distinguish evidence from assumptions and "
                "never invent sources or claim to have verified "
                "facts that the evidence does not support."
            ),
            llm=llm,
            verbose=False,
            allow_delegation=False
        )

        research_task = Task(
            description=f"""
Research topic: {topic}

Use the following DuckDuckGo web search results as your
starting evidence:

{sources_text}

Write a detailed report with these sections:

1. Title
2. Executive summary
3. Introduction
4. Key findings
5. Benefits and opportunities
6. Challenges and limitations
7. Conclusion
8. References

Requirements:
- Use clear, beginner-friendly language.
- Base factual claims on the provided search results.
- Include relevant source URLs beside the claims they support.
- Do not invent statistics, quotations, or references.
- If evidence is insufficient, state that clearly.
- Explain conflicting evidence where relevant.
- Do not claim that you opened or read full articles unless
  their content was actually provided.
""",
            expected_output=(
                "A well-structured research report with key findings, "
                "balanced analysis, a conclusion, and source URLs."
            ),
            agent=researcher
        )

        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential,
            verbose=False
        )

        with st.spinner("Writing your research report..."):
            result = crew.kickoff()

        report = str(result)

        st.success("Research report generated!")
        st.markdown(report)

        st.download_button(
            label="Download Research Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown"
        )

        with st.expander("View web search results"):
            for item in search_results:
                st.markdown(f"**{item['title']}**")
                st.write(item["snippet"])
                if item["url"]:
                    st.markdown(f"[Open source]({item['url']})")

    except Exception as error:
        st.error("The research report could not be generated.")
        st.code(f"{type(error).__name__}: {error}")
        st.info(
            "Check the Streamlit Secrets, installed package versions, "
            "Groq model availability, and deployment logs."
        )
