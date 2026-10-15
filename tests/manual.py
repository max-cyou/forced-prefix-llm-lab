from engine import Runner


def unescape(s):
    return s.replace(r"\n", "\n").replace(r"\t", "\t")


def main():
    runner = Runner()
    print("Ready. Enter q as prompt to exit.")
    print("Use \\n for newlines.\n")

    try:
        while True:
            prompt = input("Prompt: ")

            if prompt.strip().lower() in {"q", "quit", "exit"}:
                break

            system = input("System prompt (optional): ")
            prefix = input("Forced prefix: ")

            try:
                result = runner.compare(
                    unescape(system),
                    unescape(prompt),
                    unescape(prefix),
                )

                print(
                    "\n=== NORMAL ===\n",
                    result["normal_response"],
                    sep="",
                )

                print(
                    "\n=== FORCED (complete) ===\n",
                    result["forced_full_response"],
                    sep="",
                )

                print(
                    "\n=== FORCED (free continuation) ===\n",
                    result["forced_free_response"],
                    sep="",
                )

                print(
                    "\nVERIFIED:",
                    result["forced_token_ids_verified"],
                )

                for mode in ("normal", "forced"):
                    data = result[mode]

                    print(
                        f"{mode.upper()}: "
                        f"{data['tokens']} tokens, "
                        f"{data['tokens_per_second']} tok/s, "
                        f"{data['stop_reason']}"
                    )

                print()

            except (ValueError, RuntimeError) as error:
                print("Error:", error)

    except (KeyboardInterrupt, EOFError):
        print("\nExit")

    finally:
        runner.close()


if __name__ == "__main__":
    main()
