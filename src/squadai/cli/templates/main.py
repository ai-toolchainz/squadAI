#!/usr/bin/env python
from {{folder_name}}.squad import {{squad_name}}Squad


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'AI LLMs'
    }
    {{squad_name}}Squad().squad().kickoff(inputs=inputs)