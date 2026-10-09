from typing import List

class BasePipeline:

    def user_input_to_sequence(self, input_text: str) -> List[int]:
        """Convert raw user input into a numerical sequence for ML inference."""
        return []