import json
import os
import yaml
from textwrap import dedent
from typing import Optional, List, Any

from crewai import Agent, Crew, Process, Task
from crewai_tools import FileReadTool, DirectoryReadTool
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


class Job(BaseModel):
    job_id: Optional[str]
    location: Optional[str]
    title: Optional[str]
    company: Optional[str]
    description: Optional[str]
    jobProvider: Optional[str]
    link: Optional[str]
    rating: Optional[int]
    rating_description: Optional[str]
    company_rating: Optional[int]
    company_rating_description: Optional[str]


class JobResults(BaseModel):
    jobs: Optional[List[Job]]


class JobLoaderTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="Job Loader",
            description="Loads job descriptions from JSON files.",
        )

    def _run(self, **kwargs: Any) -> List[Job]:
        folder_path = kwargs.get("folder_path")
        jobs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                with open(os.path.join(folder_path, file_name), "r") as file:
                    jobs.append(json.load(file))
        return jobs


class JobLoaderAgent(Agent):
    def __init__(self, job_loader):
        super().__init__(
            role="Job Loader",
            goal="Load job descriptions from JSON files.",
            backstory="An AI agent that retrieves job postings from a folder.",
            tools=[job_loader],
            verbose=True
        )


class AgentsFactory:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def create_agent(
        self,
        agent_type: str,
        llm: Any,
        tools: Optional[List] = None,
        verbose: bool = True,
        allow_delegation: bool = False,
    ) -> Agent:
        agent_config = self.config.get(agent_type)
        if not agent_config:
            raise ValueError(f"No configuration found for {agent_type}")

        if tools is None:
            tools = []

        return Agent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            backstory=agent_config["backstory"],
            verbose=verbose,
            tools=tools,
            llm=llm,
            allow_delegation=allow_delegation,
        )


class TasksFactory:
    def __init__(self, config_path):
        self.config = load_config(config_path)

    def create_task(
        self,
        task_type: str,
        agent: Agent,
        query: Optional[str] = None,
        output_schema: Optional[str] = None,
    ):
        task_config = self.config.get(task_type)
        if not task_config:
            raise ValueError(f"No configuration found for {task_type}")

        description = task_config["description"]
        if "{query}" in description and query is not None:
            description = description.format(query=query)

        expected_output = task_config["expected_output"]
        if "{output_schema}" in expected_output and output_schema is not None:
            expected_output = expected_output.format(output_schema=output_schema)

        return Task(
            description=dedent(description),
            expected_output=dedent(expected_output),
            agent=agent,
            config=task_config,
        )


class JobPreprocessorTool(BaseTool):
    name: str = "Data Preprocessor"
    description: str = "Preprocesses data by handling missing values, removing duplicates, and encoding categorical variables."

    def _run(self, file_path: str) -> str:
        # Load the data
        with open(file_path, 'r') as file:
            job = json.loads(file.read())
            return json.dumps({"title": job["title"]})

        # # Get initial info
        # initial_shape = df.shape
        # initial_missing = df.isnull().sum().sum()
        #
        # # Handle missing values
        # df = df.dropna()  # or use df.fillna() with appropriate strategy
        #
        # # Remove duplicate entries
        # df = df.drop_duplicates()
        #
        # # Identify categorical columns
        # categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        #
        # # Convert categorical variables to numerical
        # label_encoder = LabelEncoder()
        # for col in categorical_columns:
        #     df[col] = label_encoder.fit_transform(df[col])
        #
        # # Get final info
        # final_shape = df.shape
        # final_missing = df.isnull().sum().sum()
        #
        # # Save the processed data
        # processed_file_path = os.path.join('processed_data', 'processed_data.csv')
        # df.to_csv(processed_file_path, index=False)
        #
        # return f"""
        # Data preprocessing completed:
        # - Initial shape: {initial_shape}
        # - Initial missing values: {initial_missing}
        # - Final shape: {final_shape}
        # - Final missing values: {final_missing}
        # - Categorical variables encoded: {categorical_columns}
        # - Duplicates removed
        # - Processed data saved to: {processed_file_path}
        # """

class JobSearchCrew:
    def __init__(self, query: str):
        self.query = query

    def run(self):
        os.environ['OPENAI_API_KEY'] = 'ollama'
        llm = ChatOpenAI(
            model="ollama/deepseek-r1:14b",
            base_url="http://localhost:11434/v1",
            max_tokens=32768,
        )

        # Initialize all tools needed
        #jobs_file_read_tool = FileReadTool(file_path="data/jobs.json")
        # job_loader_tool = JobLoaderTool(folder_path="/Users/miron/Dev/rag/content_creation_01/job_search_agents_01/jobs/")
        job_loader_tool = JobLoaderTool()
        jobs_directory_read_tool = DirectoryReadTool(directory="/Users/miron/Dev/rag/content_creation_01/job_search_agents_01/jobs/")
        jobs_processor_tool = JobPreprocessorTool()

        # Create the Agents
        agent_factory = AgentsFactory("configs/03-agents.yml")
        job_loader_agent = agent_factory.create_agent(
            "job_loader", tools=[job_loader_tool], llm=llm
        )
        job_search_expert_agent = agent_factory.create_agent(
            # "job_search_expert", tools=[jobs_file_read_tool], llm=llm
            "job_search_expert", tools=[jobs_directory_read_tool, jobs_processor_tool], llm=llm
        )

        # Create the Tasks
        tasks_factory = TasksFactory("configs/03-tasks.yml")
        job_load_task = tasks_factory.create_task("job_loader", job_loader_agent)
        job_search_task = tasks_factory.create_task(
            "job_search", job_search_expert_agent, query=self.query
        )

        # Assemble the Crew
        crew = Crew(
            agents=[
                job_loader_agent,
            ],
            tasks=[
                job_load_task,
            ],
            verbose=True,
            process=Process.sequential,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    print("## Welcome to Job Search Crew")
    print("-------------------------------")
    query = input(
        dedent("""
      Provide the list of characteristics for the job you are looking for: 
    """)
    )

    crew = JobSearchCrew(query)
    result = crew.run()

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