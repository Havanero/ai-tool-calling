import json
from functools import wraps
from io import StringIO
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

triaging_system_prompt = """You are a Triaging Agent. Your role is to assess
 the user's query and route it to the relevant agents. The agents available
 are:
- Data Processing Agent: Cleans, transforms, and aggregates data.
- Analysis Agent: Performs statistical, correlation, and regression analysis.
- Visualization Agent: Creates bar charts, line charts, and pie charts.

Use the send_query_to_agents tool to forward the user's query to the relevant
agents. Also, use the speak_to_user tool to get more information from the user
if needed."""

processing_system_prompt = """You are a Data Processing Agent. Your role is to
  clean, transform, and aggregate data using the following tools:
- clean_data
- transform_data
- aggregate_data"""

analysis_system_prompt = """You are an Analysis Agent. Your role is to perform
statistical, correlation, and regression analysis using the following tools:
- stat_analysis
- correlation_analysis
- regression_analysis"""

visualization_system_prompt = """You are a Visualization Agent. Your role is to
create bar charts, line charts, and pie charts using the following tools:
- create_bar_chart
- create_line_chart
- create_pie_chart"""


class SendQueryToAgents(BaseModel):
    """Sends the user query to relevant agents based on their capabilities."""

    agents: List[str]
    query: str


class CleanData(BaseModel):
    """Cleans the provided data by removing duplicates and handling missing
    values."""

    data: str = Field(
        description="The dataset to clean. Should be in a suitable "
        + "format such as JSON or CSV."
    )

    @classmethod
    def clean_data(cls, data):
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=",")
        df_deduplicated = df.drop_duplicates()
        return df_deduplicated


class TransformData(BaseModel):
    """Transforms data based on specified rules"""

    data: str = Field(
        description="The data to transform. Should be in a suitable format "
        + "such as JSON or CSV."
    )
    riles: str = Field(
        description="Transformation rules to apply, specified in a structured"
        + "format."
    )


class AggregateData(BaseModel):
    """Aggregates data by specified columns and operations."""

    data: str = Field(
        description="The data to aggregate. Should be in a suitable format such as JSON or CSV."
    )
    group_by: List[str] = Field(description="Columns to group by.")
    operations: str = Field(
        description="Aggregation operations to perform, specified in a structured format."
    )


class StatAnalysisParams(BaseModel):
    """
    Parameters for performing statistical analysis on a dataset.
    """

    data: str = Field(
        description="The dataset to analyze. Should be in a suitable format such as JSON or CSV."
    )

    @classmethod
    def stat_analysis(cls, data):
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=",")
        return df.describe()


class CorrelationAnalysisParams(BaseModel):
    """
    Parameters for calculating correlation coefficients between variables in a dataset.
    """

    data: str = Field(
        description="The dataset to analyze. Should be in a suitable format such as JSON or CSV."
    )
    variables: List[str] = Field(
        description="List of variables to calculate correlations for."
    )


class RegressionAnalysisParams(BaseModel):
    """
    Parameters for performing regression analysis on a dataset.
    """

    data: str = Field(
        description="The dataset to analyze. Should be in a suitable format such as JSON or CSV."
    )
    dependent_var: str = Field(description="The dependent variable for regression.")
    independent_vars: List[str] = Field(description="List of independent variables.")


class CreateBarChartParams(BaseModel):
    """
    Creates a bar chart from the provided data.
    """

    data: str = Field(
        description="The data for the bar chart. Should be in a suitable format such as JSON or CSV."
    )
    x: str = Field(description="Column for the x-axis.")
    y: str = Field(description="Column for the y-axis.")


class CreateLineChartParams(BaseModel):
    """
    Creates a line chart from the provided data.
    """

    data: str = Field(
        description="The data for the line chart. Should be in a suitable format such as JSON or CSV."
    )
    x: str = Field(description="Column for the x-axis.")
    y: str = Field(description="Column for the y-axis.")

    @classmethod
    def plot_line_chart(cls, data):
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=",")

        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x)

        plt.figure(figsize=(10, 6))
        plt.plot(x, y, "o", label="Data Points")
        plt.plot(x, y_fit, "-", label="Best Fit Line")
        plt.title("Line Chart with Best Fit Line")
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        plt.legend()
        plt.grid(True)
        plt.show()


class CreatePieChartParams(BaseModel):
    """
    Creates a pie chart from the provided data.
    """

    data: str = Field(
        description="The data for the pie chart. Should be in a suitable format such as JSON or CSV."
    )
    labels: str = Field(description="Column for the labels.")
    values: str = Field(description="Column for the values.")


triage_tools = [
    openai.pydantic_function_tool(SendQueryToAgents, name="send_query_to_agents")
]

preprocess_tools = [
    openai.pydantic_function_tool(factory, name=factory_name)
    for factory, factory_name in (
        (CleanData, "clean_data"),
        (TransformData, "transform_data"),
        (AggregateData, "aggregate_data"),
    )
]

analysis_tools = [
    openai.pydantic_function_tool(factory, name=factory_name)
    for factory, factory_name in (
        (StatAnalysisParams, "stat_analysis"),
        (CorrelationAnalysisParams, "correlation_analysis"),
        (RegressionAnalysisParams, "regression_analysis"),
    )
]

visualization_tools = [
    openai.pydantic_function_tool(factory, name=factory_name)
    for factory, factory_name in (
        (CreateBarChartParams, "create_bar_chart"),
        (CreateLineChartParams, "create_line_chart"),
        (CreatePieChartParams, "create_pie_chart"),
    )
]


client = OpenAI()

MODEL = "gpt-4o-2024-08-06"


def tool_dispatch(func):
    """
    A custom dispatcher that routes based on `tool_name`.
    """
    registry = {}

    @wraps(func)
    def wrapper(args, **kwargs):
        tool_name = args.get("tool_name") if isinstance(args, dict) else None
        handler = registry.get(tool_name, func)
        return handler(args, **kwargs)

    def register(tool_name):
        def decorator(handler):
            registry[tool_name] = handler
            return handler

        return decorator

    wrapper.register = register
    return wrapper


@tool_dispatch
def handle_tool_call(args):
    tool_name = args.get("tool_name", "unknown")
    raise ValueError(f"Unknown tool: {tool_name}")


# Register specific tool handlers
@handle_tool_call.register("send_query_to_agents")
def _(args):
    agents = args.get("agents", [])
    query = args.get("query", "")

    result = {"response": f"Query '{query}' sent to agents: {', '.join(agents)}"}
    return result


@handle_tool_call.register("clean_data")
def _(args):
    cleaned_df = CleanData.clean_data(args["data"])
    return {"cleaned_data": cleaned_df.to_dict()}


@handle_tool_call.register("transform_data")
def _(args):
    return {"transformed_data": "sample_transformed_data"}


@handle_tool_call.register("aggregate_data")
def _(args):
    return {"aggregated_data": "sample_aggregated_data"}


@handle_tool_call.register("stat_analysis")
def _(args):
    stats_df = StatAnalysisParams.stat_analysis(args["data"])
    return {"stats": stats_df.to_dict()}


@handle_tool_call.register("correlation_analysis")
def _(args):
    return {"correlations": "sample_correlations"}


@handle_tool_call.register("regression_analysis")
def _(args):
    return {"regression_results": "sample_regression_results"}


@handle_tool_call.register("create_bar_chart")
def _(args):
    return {"bar_chart": "sample_bar_chart"}


@handle_tool_call.register("create_line_chart")
def _(args):
    CreateLineChartParams.plot_line_chart(args["data"])
    return {"line_chart": "sample_line_chart"}


@handle_tool_call.register("create_pie_chart")
def _(args):
    return {"pie_chart": "sample_pie_chart"}


# Main function to execute tool calls
def execute_tool(tool_calls, messages):
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)
        tool_arguments["tool_name"] = tool_name
        result = handle_tool_call(tool_arguments)
        messages.append(
            {"role": "tool", "name": tool_name, "content": json.dumps(result)}
        )
    return messages


# AGENTS
def handle_data_processing_agent(query, conversation_messages):
    messages = [{"role": "system", "content": processing_system_prompt}]
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=triage_tools,
    )
    print("Agent 3 comnplete")
    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )
    execute_tool(response.choices[0].message.tool_calls, conversation_messages)


def handle_analysis_agent(query, conversation_messages):
    messages = [{"role": "system", "content": analysis_system_prompt}]
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=preprocess_tools,
    )

    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )
    execute_tool(response.choices[0].message.tool_calls, conversation_messages)


def handle_visualization_agent(query, conversation_messages):
    messages = [{"role": "system", "content": visualization_system_prompt}]
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=visualization_tools,
    )

    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )
    execute_tool(response.choices[0].message.tool_calls, conversation_messages)


# Function to handle user input and triaging
def handle_user_message(user_query, conversation_messages=[]):

    user_message = {"role": "user", "content": user_query}

    conversation_messages.append(user_message)

    messages = [{"role": "system", "content": triaging_system_prompt}]
    messages.extend(conversation_messages)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        tools=triage_tools,
    )

    conversation_messages.append(
        [tool_call.function for tool_call in response.choices[0].message.tool_calls]
    )

    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "send_query_to_agents":
            agents = json.loads(tool_call.function.arguments)["agents"]
            query = json.loads(tool_call.function.arguments)["query"]
            for agent in agents:
                if agent == "Data Processing Agent":
                    handle_data_processing_agent(query, conversation_messages)
                elif agent == "Analysis Agent":
                    handle_analysis_agent(query, conversation_messages)
                elif agent == "Visualization Agent":
                    handle_visualization_agent(query, conversation_messages)
    return conversation_messages
