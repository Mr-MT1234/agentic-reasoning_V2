import os

from .agent import Agent


class CodeAgent(Agent):
    """Agent that generates Python code from a given query using a Language Model and executes it safely in Docker."""

    def __init__(
        self, name: str = "code_agent", model=None, docker_image_name="python_executor"
    ):
        self.name = name
        self.model = model
        self.docker_image_name = docker_image_name
        self.docker_client = docker.from_env()
        self.build_docker_image()

    def build_docker_image(self):
        """Build the Docker image used for executing Python code."""
        dockerfile_content = """
        FROM python:3.11-slim
        WORKDIR /app
        """

        dockerfile_path = "temp/tools/Dockerfile"
        os.makedirs(os.path.dirname(dockerfile_path), exist_ok=True)

        with open(dockerfile_path, "w") as file:
            file.write(dockerfile_content)

        self.docker_client.images.build(
            path="temp/tools",
            dockerfile="Dockerfile",
            tag=self.docker_image_name,
            rm=True,
        )

    def generate_code(self, query: str, context: str = "") -> str:
        """Generate Python code from a given query using a Language Model and execute it in Docker."""
        prompt = f"Given the Context: {context}\n\nWrite a Python script that solves the Problem. Ensure it can be run and outputs results directly. OUTPUT ONLY CODE. Problem: {query}"
        result = self.model.generate_response_from_prompt(prompt)

        if "```python" in result:
            result = result[result.find("```python") + 9 : result.rfind("```")]
        elif "```" in result:
            result = result[result.find("```") + 3 : result.rfind("```")]

        path = "temp/tools/temp.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as file:
            file.write(result)

        return self.execute_code_in_docker(path)

    def execute_code_in_docker(self, script_path: str):
        """Run the Python script within a Docker container."""
        abs_path = os.path.abspath(script_path)

        try:
            container = self.docker_client.containers.run(
                image=self.docker_image_name,
                command=["python", "/app/temp.py"],
                volumes={abs_path: {"bind": "/app/temp.py", "mode": "ro"}},
                working_dir="/app",
                remove=True,
                stdout=True,
                stderr=True,
            )
            return container.decode("utf-8")

        except docker.errors.ContainerError as e:
            return f"Execution error: {e.stderr.decode('utf-8')}"
        except docker.errors.APIError as e:
            return f"Docker API error: {str(e)}"
        except docker.errors.DockerException as e:
            return f"Docker error: {str(e)}"

    def __call__(self, query: str, context: str = "") -> str:
        """Convenience method to directly generate and execute Python code."""
        return self.generate_code(query, context)
