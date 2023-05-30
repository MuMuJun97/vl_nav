import os
import sys


class InstructionPro(object):
    def __init__(self, instr, type):
        self.task_description = "You are a mobile agent in an indoor building."
        self.prompt = "System: {task_description}\n" \
                      "{history}\n" \
                      "Commander: {instruction}\n" \
                      "{current_images}"
        self.type = type

        if self.type == 'r2r':
            self.instr = 'Travel following the instruction, you can not ask for help. Instruction: ' \
                + instr
        else:
            raise NotImplementedError

        self.input_text = self.prompt.format(
            task_description=self.task_description,
            history='{history}',
            instruction=self.instr,
            current_images='{current_images}',
        )

    def str(self):
        return self.input_text

    def update(self):
        self.input_text

