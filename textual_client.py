import logging
import os
import re
import traceback

from textual.widgets import _markdown as _tx_md


def _always_safe_slug(text: str) -> str:
    """Return a slug made of letters, digits, underscores and hyphens — nothing else.

    Ensures it never starts with a digit so Textual accepts the resulting ID.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")
    if not slug or slug[0].isdigit():
        slug = f"h{slug}"
    return slug


_tx_md.slug = _always_safe_slug  # Override Textual's slug function

from openai import AsyncOpenAI  # noqa: E402
from textual import on, work  # noqa: E402
from textual.app import App, ComposeResult  # noqa: E402
from textual.containers import Horizontal, VerticalScroll  # noqa: E402
from textual.widgets import Footer, Header, Input, Markdown  # noqa: E402


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
        ("ctrl+r", "reset", "Reset"),
        ("ctrl+1", "set_reasoning_low", "Low"),
        ("ctrl+2", "set_reasoning_medium", "Med"),
        ("ctrl+3", "set_reasoning_high", "High"),
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
        height: 1fr;
        overflow-y: auto;
    }

    #input-container {
        height: 3;
        dock: bottom;
        margin-bottom: 3;
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
        self.reasoning_effort = "medium"

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

    async def action_set_reasoning_low(self) -> None:
        """Set reasoning effort to low."""
        self.reasoning_effort = "low"
        self.notify(
            f"Reasoning effort: {self.reasoning_effort}", severity="information"
        )

    async def action_set_reasoning_medium(self) -> None:
        """Set reasoning effort to medium."""
        self.reasoning_effort = "medium"
        self.notify(
            f"Reasoning effort: {self.reasoning_effort}", severity="information"
        )

    async def action_set_reasoning_high(self) -> None:
        """Set reasoning effort to high."""
        self.reasoning_effort = "high"
        self.notify(
            f"Reasoning effort: {self.reasoning_effort}", severity="information"
        )

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
        self.current_reasoning = Reasoning()
        self.current_output = Output()
        await chat_view.mount(self.current_reasoning)
        await chat_view.mount(self.current_output)

        # Scroll to latest
        self.current_output.scroll_visible()

        # Send prompt to model
        self.send_prompt(event.value)

    @work
    async def send_prompt(self, prompt: str) -> None:
        """Send prompt to the model and stream response."""
        reasoning_content = ""
        output_content = ""

        try:
            await self._async_send_prompt(prompt, reasoning_content, output_content)
        except Exception as e:
            self.current_output.update(f"**Error:** {e}")

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
                reasoning={"effort": self.reasoning_effort},
                stream=True,
                temperature=0.7,
                max_output_tokens=8192,
            )

            # Process stream
            reasoning_stream = None
            output_stream = None

            try:
                async for chunk in response:
                    # Handle reasoning text delta events
                    if (
                        hasattr(chunk, "type")
                        and chunk.type == "response.reasoning_text.delta"
                    ):
                        if hasattr(chunk, "delta"):
                            text = chunk.delta
                            reasoning_content += text

                            # Create stream on first delta
                            if reasoning_stream is None:
                                reasoning_stream = Markdown.get_stream(
                                    self.current_reasoning
                                )

                            await reasoning_stream.write(text)

                    # Handle output text delta events
                    elif (
                        hasattr(chunk, "type")
                        and chunk.type == "response.output_text.delta"
                    ) and hasattr(chunk, "delta"):
                        text = chunk.delta
                        output_content += text

                        # Create stream on first delta
                        if output_stream is None:
                            output_stream = Markdown.get_stream(self.current_output)

                        await output_stream.write(text)
            finally:
                # Stop streams when done
                if reasoning_stream is not None:
                    await reasoning_stream.stop()
                if output_stream is not None:
                    await output_stream.stop()

            # Save to conversation history
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(output_content)

        except Exception as e:
            trace = traceback.format_exc()
            print(trace)
            raise e  # noqa:TRY201


if __name__ == "__main__":
    app = GPTOSSClient()
    app.run()
