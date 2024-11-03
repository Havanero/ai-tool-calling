import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI
from pydantic import BaseModel


# Models
class GetDeliveryDate(BaseModel):
    order_id: str


class OrderSchema(BaseModel):
    order_id: str
    delivery_date: str


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class Config:
    MODEL_NAME = "gpt-4-0613"
    SYSTEM_PROMPT = "You are a helpful customer support assistant. Use the supplied tools to assist the user."


# Database service
class OrderService:
    @staticmethod
    def get_order_status(order_id: str) -> OrderSchema:
        """Simulates database call to get order status"""
        print(f"Fetching order {order_id} from database")
        delivery_date = "01-02-2024"
        return OrderSchema(order_id=order_id, delivery_date=delivery_date)


class ChatService:
    def __init__(self):
        self.client = OpenAI()
        self.tools = [
            openai.pydantic_function_tool(GetDeliveryDate, name="get_order_status")
        ]
        self.order_service = OrderService()
        self.messages: List[Dict[str, Any]] = []

    def initialize_chat(self):
        """Initialize chat with system message"""
        self.messages = [{"role": "system", "content": Config.SYSTEM_PROMPT}]

    def add_user_message(self, content: str):
        """Add user message to conversation"""
        self.messages.append({"role": "user", "content": content})

    def _handle_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        tool_messages = []

        if not tool_calls:
            return tool_messages

        for tool_call in tool_calls:
            try:
                if tool_call.function.name == "get_order_status":
                    args = json.loads(tool_call.function.arguments)
                    result = self.order_service.get_order_status(
                        GetDeliveryDate(**args).order_id
                    )
                    formatted_result = {
                        "order_id": result.order_id,
                        "delivery_date": result.delivery_date,
                    }
                    tool_messages.append(
                        {
                            "role": "tool",
                            "name": "get_order_status",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(formatted_result),
                        }
                    )
            except Exception as e:
                print(f"Error handling tool call: {str(e)}")
                tool_messages.append(
                    {
                        "role": "tool",
                        "name": tool_call.function.name,
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": str(e)}),
                    }
                )

        return tool_messages

    def get_response(self) -> str:
        """Get response from OpenAI API"""
        try:
            # Initial API call
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME, messages=self.messages, tools=self.tools
            )

            # Add assistant's message to conversation
            assistant_message = response.choices[0].message
            self.messages.append(assistant_message)

            # If no tool calls, return the response
            if not assistant_message.tool_calls:
                return assistant_message.content

            # Handle tool calls and get new messages
            tool_messages = self._handle_tool_calls(assistant_message.tool_calls)
            self.messages.extend(tool_messages)  # Add tool messages to conversation

            # Get final response
            final_response = self.client.chat.completions.create(
                model=Config.MODEL_NAME, messages=self.messages, tools=self.tools
            )

            return final_response.choices[0].message.content

        except Exception as e:
            return f"Error processing request: {str(e)}"


def main():
    chat_service = ChatService()
    chat_service.initialize_chat()

    chat_service.add_user_message(
        "Hi, can you tell me the delivery date for my order #12345?"
    )

    response = chat_service.get_response()
    print("Response:", response)

    chat_service.add_user_message("Thank sir")

    response = chat_service.get_response()
    print("Response:", response)


if __name__ == "__main__":
    main()
