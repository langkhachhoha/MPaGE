from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': "You are an expert in the domain of optimization heuristics helping to design heuristics that can effectively solve optimization problems."},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        prompt_content = f'''{task_prompt}
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)} \n
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content
    
    @classmethod
    def get_prompt_suggestions_only(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'

        # create prmpt content
        prompt_content = f'''
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
\n Please carefully analyze all of the above algorithms. Your task is to synthesize their ideas, 
identify recurring patterns, and point out opportunities for improvement.\n\n
Your output should be a **Suggestions** section, where you:\n
- Summarize key strengths shared across the implementations.\n 
- Identify limitations or blind spots that appear in multiple codes.\n
- Propose hybrid or improved strategies that integrate strengths and overcome shortcomings, in a feasible running time.\n\n
Output format:\n 
---\n
Suggestions:\n(Write only one propose hybrid or improved strategie that integrate strengths and overcome shortcomings here.)\n
Do not include any explanations, summaries, or new algorithms outside of this section. '''

        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function, suggestions = None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'

        if suggestions is None:
        # create prmpt content
            prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Analyze the logic of all the given code snippets carefully. Then identify the two code snippets whose logic is most different from each other
and create a new algorithm that totally different in logic and form from both of them.
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        else:
            prompt_content = f'''{task_prompt}
\n Here are some suggestions you can refer to:\n
---\n
Suggestions:\n + {suggestions} + \n
---\n\n 
Please help me create a new algorithm based on the above suggestions. 
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content
    
    @classmethod
    def get_prompt_cluster(cls, task_prompt: str, indivs: List[Function], template_function: Function, suggestions = None):
        # for indi in indivs:
        #     assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'Code {i}: \n{indi.algorithm}\n{str(indi)}\n\n'
        prompt_content = f"""
I have {len(indivs)} existing algorithms with their codes as follows: \n
{indivs_prompt}
Group them into clusters, where:
- Each cluster contains code snippets with similar logic.
- Different clusters should have maximally different logic.
- Return a JSON with a key "Group", whose value is a list of sublists.
- Each sublist contains the code indices (starting from 0) that belong to that cluster.

Return format example:
  "Group": [
    [0, 2],
    [1, 4],
    [3]
  ]
  """
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function, suggestions = None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        # create prmpt content
        if suggestions is not None:
            prompt_content = f'''{task_prompt}
\n Here are some suggestions you can refer to:\n
---\n
Suggestions:\n + {suggestions} + \n
---\n\n 
Please help me create a new algorithm based on the above suggestions.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        else:
            prompt_content = f'''{task_prompt}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one long, detail sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. You may focus on refining either the selection phase or the neighborhood search phase.
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. You may focus on refining either the selection phase or the neighborhood search phase
1. First, describe your new algorithm and main steps  in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content
    

if __name__ == '__main__':
    # Example usage
    task_prompt = "Optimize the following problem."
    template_function = Function(name="optimize", body="pass")
    prompt = EoHPrompt.get_prompt_i1(task_prompt, template_function)
    print(prompt)
