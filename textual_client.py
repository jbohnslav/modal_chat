import logging
import os
import sys
import traceback
import uuid

from urllib.parse import unquote

# ---------------------------------------------------------------------------
# Monkey‑patch Textual's global slug() so heading IDs never break validation.
# ---------------------------------------------------------------------------
import re
from textual.widgets import _markdown as _tx_md

def _always_safe_slug(text: str) -> str:
    """
    Return a slug made of letters, digits, underscores and hyphens ― nothing else.
    Ensures it never starts with a digit so Textual accepts the resulting ID.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")
    if not slug or slug[0].isdigit():
        slug = f"h{slug}"
    return slug

_tx_md.slug = _always_safe_slug  # <- overrides the slug() used inside Textual

from openai import AsyncOpenAI
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown


# Configure logging
def setup_logging():
    """Set up logging to both stdout and file."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "textual_client.log"), encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Initialize logging
logger = setup_logging()


def sanitize_content(content: str) -> str:
    """Minimally clean the streamed markdown so Textual's Markdown widget
    doesn't choke on stray percent signs.

    We **only** do two things:
    1. URL-decode any percent-encoded sequences (e.g. ``%20`` → `` ``).
    2. Remove any remaining literal ``%`` characters.

    Everything else – including markdown control characters, punctuation, and
    non-ASCII glyphs – is preserved so that the rendered output keeps its
    intended formatting.
    """
    decoded = unquote(content)
    # A lone '%' can break ID generation inside the Markdown renderer. Strip it.
    return decoded.replace("%", "")


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
        logger.debug(f"Added user message. Total messages: {len(self.messages)}")

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        logger.debug(f"Added assistant message. Total messages: {len(self.messages)}")

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
        logger.info(f"Resetting conversation with {len(self.messages)} messages")
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
        logger.info(f"Initializing GPTOSSClient with base_url: {base_url}")
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
        )
        self.conversation = ConversationManager()
        self.current_reasoning: Reasoning | None = None
        self.current_output: Output | None = None
        self.reasoning_effort = "medium"
        logger.info("GPTOSSClient initialized successfully")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="chat-container"):
            yield Output("Welcome! Type your message below to start chatting.")
        with Horizontal(id="input-container"):
            yield Input(placeholder="Type your message here...")
        yield Footer()

    async def action_reset(self) -> None:
        """Reset the conversation."""
        logger.info("Resetting conversation")
        self.conversation.reset()
        chat_container = self.query_one("#chat-container")
        chat_container.remove_children()
        await chat_container.mount(Output("Conversation reset! Ready for a new chat."))
        self.notify("Conversation reset", severity="information")
        logger.info("Conversation reset completed")

    async def action_set_reasoning_low(self) -> None:
        """Set reasoning effort to low."""
        logger.info("Setting reasoning effort to low")
        self.reasoning_effort = "low"
        self.notify(
            f"Reasoning effort: {self.reasoning_effort}", severity="information"
        )

    async def action_set_reasoning_medium(self) -> None:
        """Set reasoning effort to medium."""
        logger.info("Setting reasoning effort to medium")
        self.reasoning_effort = "medium"
        self.notify(
            f"Reasoning effort: {self.reasoning_effort}", severity="information"
        )

    async def action_set_reasoning_high(self) -> None:
        """Set reasoning effort to high."""
        logger.info("Setting reasoning effort to high")
        self.reasoning_effort = "high"
        self.notify(
            f"Reasoning effort: {self.reasoning_effort}", severity="information"
        )

    @on(Input.Submitted)
    async def on_input(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if not event.value.strip():
            return

        logger.info(
            f"User input received: {event.value[:100]}..."
        )  # Log first 100 chars
        chat_view = self.query_one("#chat-container")
        event.input.clear()

        # Add user prompt
        await chat_view.mount(Prompt(f"**You:** {event.value}"))

        # Create reasoning and output widgets
        reasoning_id = f"reasoning_{uuid.uuid4().hex[:8]}"
        output_id = f"output_{uuid.uuid4().hex[:8]}"
        logger.debug(
            f"Created widgets with IDs: reasoning={reasoning_id}, output={output_id}"
        )
        self.current_reasoning = Reasoning(id=reasoning_id)
        self.current_output = Output(id=output_id)
        await chat_view.mount(self.current_reasoning)
        await chat_view.mount(self.current_output)

        # Scroll to latest
        self.current_output.scroll_visible()

        # Send prompt to model
        self.send_prompt(event.value)

    @work
    async def send_prompt(self, prompt: str) -> None:
        """Send prompt to the model and stream response."""
        logger.info(
            f"Sending prompt to model with reasoning effort: {self.reasoning_effort}"
        )
        reasoning_content = ""
        output_content = ""

        try:
            await self._async_send_prompt(prompt, reasoning_content, output_content)
            logger.info("Prompt processing completed successfully")
        except Exception as e:
            logger.error(f"Error sending prompt: {e}")
            self.current_output.update(f"**Error:** {e}")

    async def _async_send_prompt(
        self, prompt: str, reasoning_content: str, output_content: str
    ) -> None:
        """Async method to handle streaming response."""
        try:
            # Try Responses API first
            logger.info("Attempting to use Responses API")
            # Include conversation history in the input
            conversation_context = self.conversation.get_context()
            full_input = (
                f"{conversation_context}\nUser: {prompt}"
                if conversation_context
                else prompt
            )
            logger.debug(f"Using conversation context: {bool(conversation_context)}")

            response = await self.client.responses.create(
                model="openai/gpt-oss-120b",
                instructions=self.conversation.system_prompt,
                input=full_input,
                reasoning={"effort": self.reasoning_effort},
                stream=True,
                temperature=0.7,
                max_output_tokens=8192,
            )
            logger.info("Successfully created response stream using Responses API")

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

                            await reasoning_stream.write(sanitize_content(text))

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

                        await output_stream.write(sanitize_content(text))
            finally:
                # Stop streams when done
                if reasoning_stream is not None:
                    await reasoning_stream.stop()
                if output_stream is not None:
                    await output_stream.stop()

            # Save to conversation history
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(output_content)
            logger.info(
                f"Responses API completed. Reasoning chars: {len(reasoning_content)}, Output chars: {len(output_content)}"
            )

        except AttributeError:
            # Fallback to standard chat completions API
            logger.info(
                "Responses API not available, falling back to Chat Completions API"
            )
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
                logger.info(
                    "Successfully created response stream using Chat Completions API"
                )

                # Hide reasoning widget for standard API
                self.current_reasoning.remove()

                # Process standard streaming format
                output_stream = None

                try:
                    async for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            output_content += content

                            # Create stream on first delta
                            if output_stream is None:
                                output_stream = Markdown.get_stream(self.current_output)

                            await output_stream.write(sanitize_content(content))
                finally:
                    # Stop stream when done
                    if output_stream is not None:
                        await output_stream.stop()

                # Save to conversation history
                self.conversation.add_user_message(prompt)
                self.conversation.add_assistant_message(output_content)
                logger.info(
                    f"Chat Completions API completed. Output chars: {len(output_content)}"
                )

            except Exception as e:
                logger.error(f"Error with chat completions: {e}")
                self.current_output.update(f"**Error with chat completions:** {e}")
        except Exception as e:
            # self.current_output.update(f"**Unexpected error:** {e}")
            logger.error(f"Unexpected error: {e}")
            logger.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    logger.info("Starting GPTOSSClient application")
    app = GPTOSSClient()
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Application shutdown complete")
