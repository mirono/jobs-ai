import json
import os

from typing import Optional, List, Any, Type

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError, Field

class Job(BaseModel):
    job_id: Optional[str]
    location: Optional[str]
    title: Optional[str]
    company: Optional[str]
    description: Optional[str]
    link: Optional[str]


class JobResults(BaseModel):
    jobs: Optional[List[Job]]


os.environ['OPENAI_API_KEY'] = 'ollama'
llm = ChatOpenAI(
    model="ollama/deepseek-r1:14b",
    base_url="http://localhost:11434/v1",
    max_tokens=32768,
)

query = "Cloud Architect"

class FetchJobsInput(BaseModel):
    """Input for FetchLJobs."""

    directory: str = Field(
        ..., description="The directory that hold all the job files."
    )


class FetchJobsOutput(BaseModel):
    jobs: List[Job]


class FetchJobsTool(BaseTool):
    name: str = "Fetch job descriptions from the jobs folder"
    description: str = (
        "Fetches all the job descriptions from the jobs folder and return a list of all jobs."
    )
    args_schema: Type[BaseModel] = FetchJobsInput
    return_schema: Type[BaseModel] = FetchJobsOutput

    def _run(
        self,
        directory: str,
    ) -> FetchJobsOutput:
        dir = "jobs"
        jobs = []
        for file_name in os.listdir(dir):
            if file_name.endswith(".json"):
                print(f"- {file_name}")
                with open(os.path.join(dir, file_name), "r") as file:
                    try:
                        jobs.append(Job.model_validate(json.load(file)))
                    except Exception as e:
                        print(e)
        return FetchJobsOutput(jobs=jobs)

fetch_jobs_tool = FetchJobsTool()

job_fetch_agent = Agent(
    role="Job fetch Agent",
    goal="Load job descriptions from a folder and add them to the vector database",
    verbose=True,
    allow_delegation=False,
    backstory=(
        """
        - A dedicated professional focused on extracting and processing content
            from job files.
        - You ensure that all job content is accurately loaded and added to 
            the vector database.
        - You are thorough and fact-driven, ensuring the highest quality of data.
        - You use all your tools to fetch the job files.
        """
    ),
    tools=[fetch_jobs_tool],
    llm=llm,
)

fetch_jobs_task = Task(
    description=(
        """
        Read jobs from the specified folder.
        Extract relevant information about the content of the jobs.
        Ensure that all information comes directly from the jobs loaded. 
        Do not make up any information.

        Here is the folder:

        {directory}
        """
    ),
    expected_output="""
        A structured JSON list of all the loaded jobs.
        """,
    tools=[fetch_jobs_tool],
    agent=job_fetch_agent,
)

crew = Crew(
    agents=[job_fetch_agent, ],
    tasks=[fetch_jobs_task,],
    verbose=True,
    process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

if __name__ == "__main__":
    print("## Welcome to Job Search Crew")
    print("-------------------------------")
    print(f"Provide the list of characteristics for the job you are looking for: {query}")
    result = crew.kickoff() # inputs={"directory": "./jobs"})

    print("Validating final result..")
    try:
        validated_result = JobResults.model_validate_json(result)
    except ValidationError as e:
        print(e.json())
        print("Data output validation error, trying again...")

    print("\n\n########################")
    print("## VALIDATED RESULT ")
    print("########################\n")
    print(result)