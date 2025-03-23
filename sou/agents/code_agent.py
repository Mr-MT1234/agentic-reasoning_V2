import subprocess
import litellm
import os
from .agent import Agent

class CodeAgent(Agent):
    """Agent that generates Python code from a given query using a Language Model and executes it safely in Docker."""

    def __init__(
        self, name:str="code_agent", model=None, docker_image_name="python_executor"
    ):
        self.name = name
        self.model = model
        self.docker_image_name = docker_image_name
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

        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                self.docker_image_name,
                "-f",
                dockerfile_path,
                "temp/tools",
            ],
            check=True,
        )

    def generate_code(self, query: str, context: str = "") -> str:
        """Generate Python code from a given query using a Language Model and execute it in Docker."""
        prompt = f"Given the Context: {context}\n\nWrite a Python script that solves the Problem. Ensure it can be run and outputs results directly. OUTPUT ONLY CODE. Problem: {query}"
        result = self.model.generate_response_from_prompt(prompt)

        if "```python" in result:
            result = result[
                result.find("```python") + 9 : result.rfind("```")
            ]  # extract python code
        elif "```" in result:
            result = result[
                result.find("```") + 3 : result.rfind("```")
            ]  # extract general code

        path = "temp/tools/temp.py"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as file:
            file.write(result)

        return self.execute_code_in_docker(path)

    def execute_code_in_docker(self, script_path: str):
        """Run the Python script within a Docker container."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{os.path.abspath(script_path)}:/app/temp.py",
                    self.docker_image_name,
                    "python",
                    "temp.py",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return result.stdout

        except subprocess.TimeoutExpired:
            return "Code execution timed out after 10 seconds"

        except subprocess.CalledProcessError as e:
            return f"Execution error: {e.stderr}"

    def __call__(self, query: str, context: str = "") -> str:
        """Convenience method to directly generate and execute Python code."""
        return self.generate_code(query, context)
