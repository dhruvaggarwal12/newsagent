import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# Import necessary modules from crewai
from crewai_tools import SerperDevTool
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the tool for internet searching capabilities
tool = SerperDevTool()

# Call the Gemini models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", verbose=True, temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))

# Creating a senior researcher agent with memory and verbose mode
news_researcher = Agent(
    role="Senior Researcher",
    goal='Uncover groundbreaking technologies in {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=True
)

# Creating a writer agent with custom tools responsible for writing news blog
news_writer = Agent(
    role='Writer',
    goal='Narrate compelling tech stories about {topic}',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner."
    ),
    tools=[tool],
    llm=llm,
    allow_delegation=False
)

# Research task
research_task = Task(
    description=(
        "Identify the next big trend in {topic}. Focus on identifying pros and cons and the overall narrative. Your final report should clearly articulate the key points, its market opportunities, and potential risks."
    ),
    expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
    tools=[tool],
    agent=news_researcher,
)

# Writing task with language model configuration
write_task = Task(
    description=(
        "Compose an insightful article on {topic}. Focus on the latest trends and how it's impacting the industry. This article should be easy to understand, engaging, and positive."
    ),
    expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
    tools=[tool],
    agent=news_writer,
    async_execution=False,
    output_file='new-blog-post.md'
)

# Forming the tech-focused crew with some enhanced configuration
crew = Crew(
    agents=[news_researcher, news_writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

# Streamlit app
def main():
    st.title("AI News Generation")
    
    # Input for the topic
    topic = st.text_input("Enter the topic for research", "AI in healthcare")
    
    if st.button("Generate Report and Article"):
        result = crew.kickoff(inputs={'topic': topic})
        st.success("Task execution completed!")
        st.subheader("Research Report")
        st.write(result.get(research_task))
        st.subheader("News Article")
        st.write(result.get(write_task))
        
        # Display the content of the generated markdown file (if it exists)
        if os.path.exists('new-blog-post.md'):
            with open('new-blog-post.md', 'r') as file:
                st.markdown(file.read())

if __name__ == "__main__":
    main()
