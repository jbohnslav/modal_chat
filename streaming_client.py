import asyncio
import os

from openai import AsyncOpenAI
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()


class ConversationManager:
    def __init__(self) -> None:
        self.messages: list[dict] = []
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


class StreamingUI:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
            api_key="EMPTY",
        )
        self.conversation = ConversationManager()
        self.reasoning_text = ""
        self.output_text = ""
        self.current_section = None

    def create_layout(self, user_prompt: str = "") -> Layout:
        """Create the UI layout with panels."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="prompt", size=5),
            Layout(name="reasoning", ratio=1),
            Layout(name="output", ratio=2),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(
            Panel(
                Text(
                    "🤖 GPT-OSS Streaming Client", justify="center", style="bold cyan"
                ),
                border_style="cyan",
            )
        )

        # User prompt
        prompt_content = (
            Text(f"📝 {user_prompt}", style="green")
            if user_prompt
            else Text("💭 Waiting for input...", style="dim")
        )
        layout["prompt"].update(
            Panel(
                prompt_content,
                title="[bold green]Your Prompt[/bold green]",
                border_style="green",
            )
        )

        # Reasoning section
        reasoning_content = (
            Markdown(self.reasoning_text)
            if self.reasoning_text
            else Text("🧠 Reasoning will appear here...", style="dim italic")
        )
        layout["reasoning"].update(
            Panel(
                reasoning_content,
                title="[bold yellow]Model Reasoning[/bold yellow]",
                border_style="yellow",
            )
        )

        # Output section
        output_content = (
            Markdown(self.output_text)
            if self.output_text
            else Text("💬 Response will appear here...", style="dim")
        )
        layout["output"].update(
            Panel(
                output_content,
                title="[bold blue]Final Answer[/bold blue]",
                border_style="blue",
            )
        )

        # Footer
        layout["footer"].update(
            Panel(
                Text(
                    "Commands: [bold]Ctrl+C[/bold] to exit | Type [bold]'reset'[/bold] to clear conversation | Type [bold]'exit'[/bold] to quit",
                    justify="center",
                    style="dim",
                ),
                border_style="dim",
            )
        )

        return layout

    async def stream_response(self, prompt: str, live: Live) -> None:
        """Stream the response and update the UI."""
        self.reasoning_text = ""
        self.output_text = ""
        self.current_section = None
        current_item_type = None

        try:
            # Use the Responses API format
            response = await self.client.responses.create(
                model="openai/gpt-oss-120b",
                instructions=self.conversation.system_prompt,
                input=prompt,
                stream=True,
                temperature=0.7,
                max_output_tokens=4096,
            )

            # Process stream
            async for chunk in response:
                # The Responses API streams JSON chunks
                if hasattr(chunk, "output") and chunk.output:
                    for item in chunk.output:
                        if hasattr(item, "type"):
                            current_item_type = item.type

                        if hasattr(item, "content") and item.content:
                            for content_item in item.content:
                                if hasattr(content_item, "text"):
                                    text = content_item.text

                                    # Separate reasoning from output based on item type
                                    if current_item_type == "reasoning":
                                        self.reasoning_text += text
                                    else:
                                        self.output_text += text

                                    # Update the live display
                                    live.update(self.create_layout(prompt))

            # Save to conversation history
            self.conversation.add_user_message(prompt)
            self.conversation.add_assistant_message(self.output_text)

        except AttributeError:
            # Fallback to standard chat completions API if Responses API not available
            try:
                response = await self.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": self.conversation.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=4096,
                )

                # Process standard streaming format
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        self.output_text += content

                        # Update the live display
                        live.update(self.create_layout(prompt))

                # Save to conversation history
                self.conversation.add_user_message(prompt)
                self.conversation.add_assistant_message(self.output_text)

            except Exception as e:
                console.print(f"[red]Error with chat completions: {e}[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    async def run(self) -> None:
        """Run the main interactive loop."""
        console.print("\n[bold cyan]Welcome to GPT-OSS Streaming Client![/bold cyan]\n")

        while True:
            try:
                # Get user input
                user_prompt = Prompt.ask("\n[green]You[/green]")

                # Handle special commands
                if user_prompt.lower() == "exit":
                    break
                if user_prompt.lower() == "reset":
                    self.conversation.reset()
                    console.print("[yellow]Conversation reset![/yellow]")
                    continue
                if not user_prompt.strip():
                    continue

                # Create live display and stream response
                with Live(
                    self.create_layout(user_prompt),
                    refresh_per_second=4,
                    console=console,
                ) as live:
                    await self.stream_response(user_prompt, live)

                # Show a separator after each interaction
                console.print("\n")
                console.rule(style="dim")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

        console.print("\n[cyan]Goodbye! 👋[/cyan]\n")


async def main() -> None:
    ui = StreamingUI()
    await ui.run()


if __name__ == "__main__":
    asyncio.run(main())
