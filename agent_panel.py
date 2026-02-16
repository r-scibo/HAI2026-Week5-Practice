import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import Optional
from movie_tool import get_tools, query_movie_db
from chart_tool import get_chart_tool, validate_chart
from context_compaction import compact_if_needed


# ── State ──

DEFAULT_STATE = {
    "agent_phase": "idle",
    "agent_events": [],
    "agent_messages": [],
    "agent_tools": [],
    "agent_df": None,
    "agent_chart_specs": [],
    "agent_pending_message": None,
    "agent_approval_gen": 0,
}

def get_state(key):
    return st.session_state.get(key, DEFAULT_STATE[key])

def set_state(key, value):
    st.session_state[key] = value

def restart_agent(user_question, filtered_df, show_chart=False):
    set_state("agent_phase", "thinking")
    set_state("agent_events", [])
    set_state("agent_chart_specs", [])
    set_state("agent_pending_message", None)

    tools = get_tools(filtered_df)
    system_content = "You are a data analyst with access to a tool that executes Python code on a movie database."

    if show_chart:
        tools.append(get_chart_tool())
        system_content += " After computing the data, create a chart using a Vega-Lite specification."

    set_state("agent_messages", [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question},
    ])
    set_state("agent_tools", tools)
    set_state("agent_df", filtered_df)


# ── Logic ──

def run_step(client):
    phase = get_state("agent_phase")
    messages = get_state("agent_messages")

    tools = get_state("agent_tools")
    compacted, _, _ = compact_if_needed(client, messages, tools)
    if compacted:
        get_state("agent_events").append({
            "type": "compaction",
            "message": "Context was compacted to stay within model limits.",
        })

    if phase == "thinking":
        class Reasoning(BaseModel):
            reason: str = Field(description="Your reasoning about what you know so far and what to do next")
            use_tool: bool = Field(description="True if you need to run code or create a chart, False if you can give the final answer")
            answer: Optional[str] = Field(default=None, description="Your final answer in one short paragraph. Only provide when use_tool is False.")

        response = client.chat.completions.parse(
            model="gpt-4o-mini", messages=messages, response_format=Reasoning,
        )
        reasoning = response.choices[0].message.parsed
        messages.append({"role": "assistant", "content": reasoning.reason})

        if reasoning.use_tool:
            get_state("agent_events").append({"type": "thought", "thought": reasoning.reason})
            set_state("agent_phase", "acting")
        else:
            get_state("agent_events").append({"type": "answer", "thought": reasoning.reason, "answer": reasoning.answer})
            set_state("agent_phase", "done")

    elif phase == "acting":
        tools = get_state("agent_tools")

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools, parallel_tool_calls=False,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            set_state("agent_phase", "done")
            return

        set_state("agent_pending_message", msg)
        set_state("agent_approval_gen", get_state("agent_approval_gen") + 1)
        set_state("agent_phase", "awaiting_approval")

def execute_pending_tools(edited_args=None):
    messages = get_state("agent_messages")
    df = get_state("agent_df")
    pending_msg = get_state("agent_pending_message")

    messages.append(pending_msg)
    for i, tc in enumerate(pending_msg.tool_calls):
        args = json.loads(tc.function.arguments)

        # Use edited code/spec if the user modified it
        if edited_args and i in edited_args:
            if tc.function.name == "QueryMovieDB":
                args["code"] = edited_args[i]
            elif tc.function.name == "CreateChart":
                args["vega_lite_spec"] = edited_args[i]

        if tc.function.name == "QueryMovieDB":
            result = query_movie_db(args["code"], df)
            get_state("agent_events").append({
                "type": "action", "name": tc.function.name,
                "code": args["code"], "result": result,
            })
        elif tc.function.name == "CreateChart":
            spec, result = validate_chart(args["vega_lite_spec"])
            if spec:
                get_state("agent_chart_specs").append(spec)
            get_state("agent_events").append({
                "type": "chart", "name": tc.function.name,
                "spec_str": args["vega_lite_spec"], "result": result,
            })

        messages.append({"role": "tool", "content": result, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")

def reject_pending_tools(feedback):
    messages = get_state("agent_messages")
    pending_msg = get_state("agent_pending_message")

    rejection_msg = "User rejected this action."
    if feedback:
        rejection_msg += f" User feedback: {feedback}"
    else:
        rejection_msg += " Try a different approach."

    messages.append(pending_msg)
    for tc in pending_msg.tool_calls:
        get_state("agent_events").append({
            "type": "rejected", "name": tc.function.name,
            "feedback": feedback,
        })
        messages.append({"role": "tool", "content": rejection_msg, "tool_call_id": tc.id})

    set_state("agent_pending_message", None)
    set_state("agent_phase", "thinking")


# ── Rendering ──

def render_events():
    for event in get_state("agent_events"):
        if event["type"] == "thought":
            st.markdown(f"**Thought:** {event['thought']}")
        elif event["type"] == "action":
            st.markdown(f"**Action:** `{event['name']}`")
            st.code(event["code"], language="python")
            st.markdown("**Observation:**")
            st.code(event["result"], language="text")
            st.divider()
        elif event["type"] == "chart":
            st.markdown(f"**Action:** `{event['name']}`")
            st.code(event["spec_str"], language="json")
            st.markdown("**Observation:**")
            st.code(event["result"], language="text")
            st.divider()
        elif event["type"] == "rejected":
            st.markdown(f"**Rejected:** `{event['name']}`")
            if event.get("feedback"):
                st.text(f"Feedback: {event['feedback']}")
            st.divider()
        elif event["type"] == "compaction":
            st.info(event["message"])
        elif event["type"] == "answer":
            st.markdown(f"**Thought:** {event['thought']}")

def render_pending_approval():
    st.warning("The agent wants to perform the following action:")
    edited_args = {}
    gen = get_state("agent_approval_gen")
    for i, tc in enumerate(get_state("agent_pending_message").tool_calls):
        args = json.loads(tc.function.arguments)
        st.markdown(f"**Tool:** `{tc.function.name}`")
        if tc.function.name == "QueryMovieDB":
            edited = st.text_area(
                "Code", value=args["code"], height=150, key=f"edit_code_{gen}_{i}",
            )
            edited_args[i] = edited
        elif tc.function.name == "CreateChart":
            edited = st.text_area(
                "Vega-Lite Spec", value=args["vega_lite_spec"], height=200, key=f"edit_spec_{gen}_{i}",
            )
            edited_args[i] = edited

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        approved = st.button("Approve", type="primary", use_container_width=True)
    with btn_col2:
        rejected = st.button("Reject", use_container_width=True)
    return approved, rejected, edited_args

def render_pending_feedback():
    feedback = st.text_input(
        "Why are you rejecting? Tell the agent what to do instead:",
        key="reject_feedback",
    )
    submitted = st.button("Submit Rejection", use_container_width=True)
    return submitted, feedback

def render_panel():
    st.subheader("Analysis Results")
    container = st.container(height=600)
    actions = {}
    with container:
        phase = get_state("agent_phase")

        if phase == "idle":
            st.info("Enter a question and click 'Analyze' to see results.")

        elif phase in ("thinking", "acting"):
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            st.spinner("Agent is thinking...")

        elif phase == "awaiting_approval":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            approved, rejected, edited_args = render_pending_approval()
            actions = {"approved": approved, "rejected": rejected, "edited_args": edited_args}

        elif phase == "awaiting_feedback":
            with st.expander("Agent Reasoning Trace", expanded=True):
                render_events()
            submitted, feedback = render_pending_feedback()
            actions = {"submitted": submitted, "feedback": feedback}

        elif phase == "done":
            with st.expander("Agent Reasoning Trace", expanded=False):
                render_events()
            events = get_state("agent_events")
            if events and events[-1].get("answer"):
                st.write("**Answer:**")
                st.write(events[-1]["answer"])
            for spec in get_state("agent_chart_specs"):
                st.vega_lite_chart(spec, use_container_width=True)

    return actions


# ── Lifecycle ──

def agent_panel(client, analyze_button, user_question, filtered_df, show_chart=False):
    # Phases: idle -> thinking <-> acting -> awaiting_approval -> thinking ... -> done
    #                                     -> awaiting_feedback -> thinking ... -> done
    if analyze_button and user_question:
        restart_agent(user_question, filtered_df, show_chart)

    actions = render_panel()

    phase = get_state("agent_phase")
    if phase in ("thinking", "acting"):
        run_step(client)
        st.rerun()
    elif phase == "awaiting_approval":
        if actions.get("approved"):
            execute_pending_tools(actions.get("edited_args"))
            st.rerun()
        elif actions.get("rejected"):
            set_state("agent_phase", "awaiting_feedback")
            st.rerun()
    elif phase == "awaiting_feedback" and actions.get("submitted"):
        reject_pending_tools(actions.get("feedback", ""))
        st.rerun()
