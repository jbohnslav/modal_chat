import asyncio
import os
import uuid

from openai import AsyncOpenAI
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown


class Prompt(Markdown):
    """User prompt widget."""


class Reasoning(Markdown):
    """Model reasoning widget."""

    BORDER_TITLE = "🧠 Reasoning"


class Output(Markdown):
    """Model output widget."""

    BORDER_TITLE = "💬 Response"


class ConversationManager:
    """Manages conversation history."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.system_prompt = "You are a helpful assistant."

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def get_context(self) -> str:
        """Format conversation history for the API."""
        context = ""
        for msg in self.messages:
            if msg["role"] == "user":
                context += f"\nUser: {msg['content']}"
            elif msg["role"] == "assistant":
                context += f"\nAssistant: {msg['content']}"
        return context.strip()

    def reset(self) -> None:
        self.messages = []


class GPTOSSClient(App):
    """A Textual app for interacting with GPT-OSS models."""

    AUTO_FOCUS = "Input"
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+r", "reset", "Reset conversation"),
    ]

    CSS = """
    Prompt {
        background: $primary 10%;
        color: $text;
        margin: 1;
        margin-right: 4;
        padding: 1 2 0 2;
    }

    Reasoning {
        border: wide $warning;
        background: $warning 10%;
        color: $text;
        margin: 1;
        margin-left: 4;
        padding: 1 2 0 2;
        height: auto;
        max-height: 50%;
    }

    Output {
        border: wide $success;
        background: $success 10%;
        color: $text;
        margin: 1;
        margin-left: 4;
        padding: 1 2 0 2;
        height: auto;
    }

    #chat-container {
        height: 100%;
        overflow-y: auto;
    }

    #input-container {
        height: 3;
        dock: bottom;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
        )
        self.conversation = ConversationManager()
        self.current_reasoning: Reasoning | None = None
        self.current_output: Output | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="chat-container"):
            yield Output("Welcome! Type your message below to start chatting.")
        with Horizontal(id="input-container"):
            yield Input(placeholder="Type your message here...")
        yield Footer()

    async def action_reset(self) -> None:
        """Reset the conversation."""
        self.conversation.reset()
        chat_container = self.query_one("#chat-container")
        chat_container.remove_children()
        await chat_container.mount(Output("Conversation reset! Ready for a new chat."))
        self.notify("Conversation reset", severity="information")

    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if not event.value.strip():
            return

        chat_view = self.query_one("#chat-container")
        event.input.clear()

        # Add user prompt
        await chat_view.mount(Prompt(f"**You:** {event.value}"))

        # Create reasoning and output widgets
        reasoning_id = f"reasoning_{uuid.uuid4().hex[:8]}"
        output_id = f"output_{uuid.uuid4().hex[:8]}"
        self.current_reasoning = Reasoning(id=reasoning_id)
        self.current_output = Output(id=output_id)
        await chat_view.mount(self.current_reasoning)
        await chat_view.mount(self.current_output)

        # Scroll to latest
        self.current_output.scroll_visible()

        # Send prompt to model
        self.send_prompt(event.value)

    @work(thread=True)
    def send_prompt(self, prompt: str) -> None:
        """Send prompt to the model and stream response."""
        reasoning_content = ""
        output_content = ""

        try:
            # Run async code in thread
            asyncio.run(
                self._async_send_prompt(prompt, reasoning_content, output_content)
            )
        except Exception as e:
            self.call_from_thread(
                self.current_output.update,
                f"**Error:** {e}",
            )

    async def _async_send_prompt(
        self, prompt: str, reasoning_content: str, output_content: str
    ) -> None:
        """Async method to handle streaming response."""
        try:
            # Try Responses API first
            # Include conversation history in the input
            conversation_context = self.conversation.get_context()
            full_input = (
                f"{conversation_context}\nUser: {prompt}"
                if conversation_context
                else prompt
            )

            response = await self.client.responses.create(
                model="openai/gpt-oss-120b",
                instructions=self.conversation.system_prompt,
                input=full_input,
                stream=True,
                temperature=0.7,
                max_output_tokens=4096,
            )

            # Process stream
            async for chunk in response:
                # Handle reasoning text delta events
                if (
                    hasattr(chunk, "type")
                    and chunk.type == "response.reasoning_text.delta"
                ):
                    if hasattr(chunk, "delta"):
                        text = chunk.delta
                        reasoning_content += text
                        self.call_from_thread(
                            self.current_reasoning.update,
                            reasoning_content or "*Thinking...*",
                        )

                # Handle output text delta events
                elif (
                    hasattr(chunk, "type")
                    and chunk.type == "response.output_text.delta"
                ) and hasattr(chunk, "delta"):
                    text = chunk.delta
                    output_content += text
                    self.call_from_thread(
                        self.current_output.update,
                        output_content or "*Generating response...*",
                    )

            # Save to conversation history
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(output_content)

        except AttributeError:
            # Fallback to standard chat completions API
            try:
                # Build messages from conversation history
                messages = [
                    {"role": "system", "content": self.conversation.system_prompt}
                ]
                messages.extend(self.conversation.messages)
                messages.append({"role": "user", "content": prompt})

                response = await self.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=4096,
                )

                # Hide reasoning widget for standard API
                self.call_from_thread(self.current_reasoning.remove)

                # Process standard streaming format
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        output_content += content
                        self.call_from_thread(
                            self.current_output.update,
                            output_content or "*Generating response...*",
                        )

                # Save to conversation history
                self.conversation.add_user_message(prompt)
                self.conversation.add_assistant_message(output_content)

            except Exception as e:
                self.call_from_thread(
                    self.current_output.update,
                    f"**Error with chat completions:** {e}",
                )
        except Exception as e:
            self.call_from_thread(
                self.current_output.update,
                f"**Unexpected error:** {e}",
            )


if __name__ == "__main__":
    app = GPTOSSClient()
    app.run()
