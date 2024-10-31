# Example showcasing structured output with function/tool calling using pydantic

### The example here has been changed from the original structure output using json to a class based on pydantic

You can learn more on tool/functional calling here https://platform.openai.com/docs/guides/function-calling

The main take-away is this:

```
import openai
from pydantic import BaseModel, Field

class SendQueryToAgents(BaseModel):
    """Sends the user query to relevant agents based on their capabilities."""

    agents: List[str]
    query: str


print(openai.pydantic_function_tool(SendQueryToAgents, name="send_query_to_agents"))

{
  "type": "function",
  "function": {
    "name": "send_query_to_agents",
    "strict": true,
    "parameters": {
      "description": "Sends the user query to relevant agents based on their capabilities.",
      "properties": {
        "agents": {
          "items": {
            "type": "string"
          },
          "title": "Agents",
          "type": "array"
        },
        "query": {
          "title": "Query",
          "type": "string"
        }
      },
      "required": [
        "agents",
        "query"
      ],
      "title": "SendQueryToAgents",
      "type": "object",
      "additionalProperties": false
    },
    "description": "Sends the user query to relevant agents based on their capabilities."
  }
}
```
