# import os

# from openai import OpenAI
# from deepeval.models import DeepEvalBaseLLM

# class GroqJudge(DeepEvalBaseLLM):

#     def __init__(self):
#         self.model_name = "openai/gpt-oss-20b"

#         self.client = OpenAI(
#             api_key=os.getenv("GROQ_API_KEY"),
#             base_url="https://api.groq.com/openai/v1",
#         )

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0,
#             response_format={"type": "json_object"},
#             max_tokens=256,
#         )

#         return response.choices[0].message.content

#     async def a_generate(self, prompt: str) -> str:
#         return self.generate(prompt)

#     def get_model_name(self):
#         return self.model_name


# import os
# import asyncio
# from openai import OpenAI, AsyncOpenAI
# from deepeval.models import DeepEvalBaseLLM

# class GroqJudge(DeepEvalBaseLLM):

#     def __init__(self):
#         # CHANGED: Swapped to a blazing fast, instruction-following 8B model
#         self.model_name = "openai/gpt-oss-20b" 

#         # Synchronous client for regular evaluation iterations
#         self.client = OpenAI(
#             api_key=os.getenv("GROQ_API_KEY"),
#             base_url="https://api.groq.com/openai/v1",
#         )
        
#         # ADDED: Async client to natively support DeepEval's parallel evaluations
#         self.async_client = AsyncOpenAI(
#             api_key=os.getenv("GROQ_API_KEY"),
#             base_url="https://api.groq.com/openai/v1",
#         )

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             response_format={"type": "json_object"},
#             max_tokens=256,
#         )
#         return response.choices[0].message.content

#     # CHANGED: Rewritten to use AsyncOpenAI so DeepEval can run concurrently on your CPU
#     async def a_generate(self, prompt: str) -> str:
#         response = await self.async_client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             # response_format={"type": "json_object"},
#             max_tokens=256,
#         )
#         return response.choices[0].message.content

#     def get_model_name(self):
#         return self.model_name

# import os
# import json

# from pydantic import BaseModel
# from openai import OpenAI, AsyncOpenAI
# from deepeval.models import DeepEvalBaseLLM


# class GroqJudge(DeepEvalBaseLLM):

#     def __init__(self):

#         self.model_name = "openai/gpt-oss-20b"

#         api_key = os.getenv("GROQ_API_KEY")

#         if not api_key:
#             raise ValueError(
#                 "GROQ_API_KEY is not configured"
#             )

#         self.client = OpenAI(
#             api_key=api_key,
#             base_url="https://api.groq.com/openai/v1",
#         )

#         self.async_client = AsyncOpenAI(
#             api_key=api_key,
#             base_url="https://api.groq.com/openai/v1",
#         )

#     def load_model(self):
#         return self.client

#     # ========================================================
#     # NORMAL GENERATION
#     # ========================================================

#     def generate(self, prompt: str) -> str:

#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             temperature=0,
#             max_tokens=1056,
#         )
#         print(response.choices[0].message.content)
#         return response.choices[0].message.content

#     async def a_generate(self, prompt: str) -> str:

#         response = await self.async_client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 }
#             ],
#             temperature=0,
#             max_tokens=1056,
#         )
#         print(response.choices[0].message.content)
#         return response.choices[0].message.content

#     # ========================================================
#     # STRUCTURED GENERATION
#     # ========================================================

#     def _build_system_prompt(self, schema):

#         schema_json = json.dumps(
#             schema.model_json_schema(),
#             indent=2
#         )

#         return f"""
# You are an evaluation judge.

# Your task is to evaluate the input provided by the user.

# You MUST return ONLY one valid JSON object.

# Do NOT return:
# - markdown
# - ```json
# - explanations outside JSON
# - empty output

# The JSON MUST follow this exact schema:

# {schema_json}

# Every required field in the schema MUST be present.

# Return only the JSON object.
# """

#     # ========================================================
#     # SYNC STRUCTURED GENERATION
#     # ========================================================

#     def generate_with_schema(
#         self,
#         prompt: str,
#         schema: BaseModel,
#     ) -> BaseModel:

#         system_prompt = self._build_system_prompt(
#             schema
#         )

#         response = self.client.chat.completions.create(

#             model=self.model_name,

#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 },
#             ],

#             temperature=0,

#             max_tokens=1056,

#             response_format={
#                 "type": "json_object"
#             },
#         )

#         content = response.choices[0].message.content

#         if not content:
#             raise ValueError(
#                 "Groq returned empty response"
#             )

#         # Remove accidental markdown fences
#         content = content.strip()

#         if content.startswith("```json"):
#             content = content[7:]

#         if content.startswith("```"):
#             content = content[3:]

#         if content.endswith("```"):
#             content = content[:-3]

#         content = content.strip()

#         # Validate against DeepEval's schema
#         return schema.model_validate_json(
#             content
#         )

#     # ========================================================
#     # ASYNC STRUCTURED GENERATION
#     # ========================================================

#     async def a_generate_with_schema(
#         self,
#         prompt: str,
#         schema: BaseModel,
#     ) -> BaseModel:

#         system_prompt = self._build_system_prompt(
#             schema
#         )

#         response = await self.async_client.chat.completions.create(

#             model=self.model_name,

#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 },
#             ],

#             temperature=0,

#             max_tokens=1056,

#             response_format={
#                 "type": "json_object"
#             },
#         )

#         content = response.choices[0].message.content

#         if not content:
#             raise ValueError(
#                 "Groq returned empty response"
#             )

#         content = content.strip()

#         if content.startswith("```json"):
#             content = content[7:]

#         if content.startswith("```"):
#             content = content[3:]

#         if content.endswith("```"):
#             content = content[:-3]

#         content = content.strip()

#         return schema.model_validate_json(
#             content
#         )

#     def get_model_name(self):
#         return self.model_name


import os
import json

from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM


class OpenRouterJudge(DeepEvalBaseLLM):

    def __init__(self):

        # Choose your judge model here
        self.model_name = "minimax/minimax-m3:free"

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def load_model(self):
        return self.client

    # ========================================================
    # NORMAL GENERATION
    # ========================================================

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=3000,
        )

        content = response.choices[0].message.content

        print(content)

        return content

    async def a_generate(self, prompt: str) -> str:

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=3000,
        )

        content = response.choices[0].message.content

        print(content)

        return content

    # ========================================================
    # STRUCTURED GENERATION
    # ========================================================

    def _build_system_prompt(self, schema):

        schema_json = json.dumps(
            schema.model_json_schema(),
            indent=2
        )
        return f"""
You are an evaluation judge.

Evaluate the input provided by the user.

Return ONLY one valid JSON object.

The JSON MUST strictly follow this exact schema:

{schema_json}

Every required field MUST be present.

Keep reasoning concise.
If a reason/explanation field exists, use at most one short sentence.
Do not include unnecessary analysis.

Return only the JSON object.
# Do NOT return:
# - markdown
# - ```json
# - explanations outside JSON
# - empty output
"""

#         return f"""
# You are an evaluation judge.

# Your task is to evaluate the input provided by the user.

# You MUST return ONLY one valid JSON object.

# Do NOT return:
# - markdown
# - ```json
# - explanations outside JSON
# - empty output

# The JSON MUST follow this exact schema:

# {schema_json}

# Every required field in the schema MUST be present.

# Return only the JSON object.
# """

    # ========================================================
    # SYNC STRUCTURED GENERATION
    # ========================================================

    def generate_with_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:

        system_prompt = self._build_system_prompt(schema)

        response = self.client.chat.completions.create(

            model=self.model_name,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0,

            max_tokens=3000,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "OpenRouter returned empty response"
            )

        content = content.strip()

        # Remove accidental markdown fences
        if content.startswith("```json"):
            content = content[7:]

        if content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        return schema.model_validate_json(
            content
        )

    # ========================================================
    # ASYNC STRUCTURED GENERATION
    # ========================================================
    # async def a_generate_with_schema(
    #     self,
    #     prompt: str,
    #     schema: BaseModel,
    # ) -> BaseModel:

    #     system_prompt = self._build_system_prompt(schema)

    #     for attempt in range(4):

    #         try:
    #             response = await self.async_client.chat.completions.create(
    #                 model=self.model_name,

    #                 messages=[
    #                     {
    #                         "role": "system",
    #                         "content": system_prompt,
    #                     },
    #                     {
    #                         "role": "user",
    #                         "content": prompt,
    #                     },
    #                 ],

    #                 temperature=0,
    #                 max_tokens=1024,

    #                 response_format={
    #                     "type": "json_object"
    #                 },
    #             )

    #             content = response.choices[0].message.content

    #             print("\nRAW JUDGE OUTPUT:")
    #             print(repr(content))

    #             if not content:
    #                 raise ValueError("Empty response")

    #             content = content.strip()

    #             if content.startswith("```json"):
    #                 content = content[7:]

    #             if content.startswith("```"):
    #                 content = content[3:]

    #             if content.endswith("```"):
    #                 content = content[:-3]

    #             content = content.strip()

    #             return schema.model_validate_json(content)

    #         except Exception as e:

    #             print(
    #                 f"\nJudge attempt {attempt + 1}/4 failed:"
    #             )
    #             print(e)

    #             if attempt == 3:
    #                 raise

    #             # await asyncio.sleep(2 ** attempt)    
    async def a_generate_with_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:

        system_prompt = self._build_system_prompt(schema)

        response = await self.async_client.chat.completions.create(

            model=self.model_name,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            temperature=0,

            max_tokens=3000,

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }
        )

        content = response.choices[0].message.content
        print(content)

        if not content:
            raise ValueError(
                "OpenRouter returned empty response"
            )

        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]

        if content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        return schema.model_validate_json(
            content
        )

    def get_model_name(self):
        return self.model_name


# ================================================================
# GROQ JUDGE — Uses Groq API (remote, high rate limits, no local LLM)
# ================================================================

class GroqJudge(DeepEvalBaseLLM):
    """
    DeepEval-compatible LLM judge using Groq API.
    - Runs remotely via Groq (fast, high RPM/TPM allowances).
    - Default model: qwen/qwen3.8-27b.
    - Structured output using JSON mode + schema validation.
    - Retry logic with exponential backoff.
    """

    def __init__(
        self,
        model_name: str = "qwen/qwen3.8-27b",
        delay_between_calls: float = 1.0,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.delay = delay_between_calls
        self.max_retries = max_retries

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured in .env")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    @staticmethod
    def _clean_json_fences(content: str) -> str:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def _build_system_prompt(self, schema):
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        return f"""You are an evaluation judge.
Your task is to evaluate the input provided by the user.
You MUST return ONLY one valid JSON object.
Do NOT return the schema definition itself or wrap inside a 'properties' key.
Directly return the JSON object with the required property fields.
Do NOT include markdown, ```json fences, or text outside the JSON.

The JSON MUST strictly follow this exact schema:
{schema_json}

Every required field in the schema MUST be present. Return only the JSON object."""

    @staticmethod
    def _parse_and_validate(content: str, schema: BaseModel) -> BaseModel:
        data = json.loads(content)
        if isinstance(data, dict):
            # If the LLM echoed schema metadata wrapping the actual fields
            if "properties" in data and isinstance(data["properties"], dict):
                data = data["properties"]
        return schema.model_validate(data)

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048,
                )
                time.sleep(self.delay)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
        raise last_error

    async def a_generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=2048,
                )
                await asyncio.sleep(self.delay)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
        raise last_error

    def generate_with_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
        system_prompt = self._build_system_prompt(schema)
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Groq returned empty response")
                content = self._clean_json_fences(content)
                result = self._parse_and_validate(content, schema)
                time.sleep(self.delay)
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
        raise last_error

    async def a_generate_with_schema(self, prompt: str, schema: BaseModel) -> BaseModel:
        system_prompt = self._build_system_prompt(schema)
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Groq returned empty response")
                content = self._clean_json_fences(content)
                result = self._parse_and_validate(content, schema)
                await asyncio.sleep(self.delay)
                return result
            except Exception as e:
                last_error = e
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
        raise last_error


# ================================================================
# GEMINI JUDGE — Uses Google Gemini API (remote, no local LLM)
# ================================================================

import time
import asyncio
from google import genai
from google.genai import types as genai_types


class GeminiJudge(DeepEvalBaseLLM):
    """
    DeepEval-compatible LLM judge using Google Gemini API (google-genai SDK).

    - No local LLM needed — runs entirely via Google's API.
    - Uses gemini-3.6-flash by default (free tier).
    - Built-in retry with exponential backoff to handle transient errors.
    - Inter-call delay to stay well under rate limits.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        delay_between_calls: float = 5.0,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.delay = delay_between_calls
        self.max_retries = max_retries

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not configured in .env"
            )

        self.client = genai.Client(api_key=api_key)

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _clean_json_fences(content: str) -> str:
        """Strip markdown code fences from LLM output."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def _retry_delay(self, attempt: int):
        """Exponential backoff: 16s, 32s, 64s, ..."""
        wait = 2 ** (attempt + 4)
        print(f"  [GeminiJudge] Retrying in {wait}s (attempt {attempt + 1}/{self.max_retries})...")
        time.sleep(wait)

    async def _async_retry_delay(self, attempt: int):
        """Async exponential backoff."""
        wait = 2 ** (attempt + 4)
        print(f"  [GeminiJudge] Retrying in {wait}s (attempt {attempt + 1}/{self.max_retries})...")
        await asyncio.sleep(wait)

    # ========================================================
    # PLAIN TEXT GENERATION
    # ========================================================

    def generate(self, prompt: str) -> str:

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=4096,
                    ),
                )
                content = response.text
                time.sleep(self.delay)
                return content

            except Exception as e:
                last_error = e
                print(f"  [GeminiJudge] generate() error: {e}")
                if attempt < self.max_retries - 1:
                    self._retry_delay(attempt)

        raise last_error

    async def a_generate(self, prompt: str) -> str:

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=4096,
                    ),
                )
                content = response.text
                await asyncio.sleep(self.delay)
                return content

            except Exception as e:
                last_error = e
                print(f"  [GeminiJudge] a_generate() error: {e}")
                if attempt < self.max_retries - 1:
                    await self._async_retry_delay(attempt)

        raise last_error

    # ========================================================
    # STRUCTURED (SCHEMA) GENERATION
    # ========================================================

    def _build_schema_prompt(self, schema, user_prompt: str) -> str:
        """Build a single prompt that includes schema instructions + user query."""
        schema_json = json.dumps(
            schema.model_json_schema(), indent=2
        )
        return f"""You are an evaluation judge.

Evaluate the input provided below and return ONLY one valid JSON object.

RULES:
- Return ONLY the JSON object, nothing else.
- Do NOT wrap in markdown code fences.
- Do NOT include any explanations outside the JSON.
- Every required field in the schema MUST be present.
- Keep any "reason" field to one short sentence.

The JSON MUST follow this exact schema:

{schema_json}

INPUT TO EVALUATE:

{user_prompt}"""

    def generate_with_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:

        full_prompt = self._build_schema_prompt(schema, prompt)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    ),
                )

                content = response.text
                content = self._clean_json_fences(content)

                result = schema.model_validate_json(content)
                time.sleep(self.delay)
                return result

            except Exception as e:
                last_error = e
                print(f"  [GeminiJudge] generate_with_schema() error: {e}")
                if attempt < self.max_retries - 1:
                    self._retry_delay(attempt)

        raise last_error

    async def a_generate_with_schema(
        self,
        prompt: str,
        schema: BaseModel,
    ) -> BaseModel:

        full_prompt = self._build_schema_prompt(schema, prompt)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    ),
                )

                content = response.text
                content = self._clean_json_fences(content)

                result = schema.model_validate_json(content)
                await asyncio.sleep(self.delay)
                return result

            except Exception as e:
                last_error = e
                print(f"  [GeminiJudge] a_generate_with_schema() error: {e}")
                if attempt < self.max_retries - 1:
                    await self._async_retry_delay(attempt)

        raise last_error


# import os
# import json
# import time
# import asyncio

# from pydantic import BaseModel
# from openai import OpenAI, AsyncOpenAI
# from deepeval.models import DeepEvalBaseLLM


# class OpenRouterJudge(DeepEvalBaseLLM):
#     """
#     Free-tier friendly judge for DeepEval, using OpenRouter's :free models.

#     Defaults to a non-reasoning instruct model so no tokens are wasted on
#     hidden chain-of-thought (that's what was eating your max_tokens budget
#     with Nemotron/gpt-oss). Auto-escalates max_tokens on truncation instead
#     of failing outright.

#     Good free OpenRouter models to try if one is rate-limited/down:
#       - meta-llama/llama-3.3-70b-instruct:free
#       - qwen/qwen-2.5-72b-instruct:free
#       - google/gemini-2.0-flash-exp:free
#       - mistralai/mistral-small-3.1-24b-instruct:free

#     Free models share a small shared rate limit (roughly 20 req/min,
#     ~50 req/day per account without any credits, ~1000/day if you've
#     ever added $10 credit once). If you hit 429s constantly, either
#     switch to a different :free model above, or slow down DeepEval's
#     concurrency (evaluate(..., max_concurrent=1)).
#     """

#     def __init__(self, model_name: str = "openai/gpt-oss-20b"):

#         self.model_name = model_name

#         api_key = os.getenv("OPENROUTER_API_KEY")

#         if not api_key:
#             raise ValueError("OPENROUTER_API_KEY is not configured")

#         self.client = OpenAI(
#             api_key=api_key,
#             base_url="https://openrouter.ai/api/v1",
#         )

#         self.async_client = AsyncOpenAI(
#             api_key=api_key,
#             base_url="https://openrouter.ai/api/v1",
#         )

#     def load_model(self):
#         return self.client

#     # ========================================================
#     # PLAIN GENERATION
#     # ========================================================

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             max_tokens=1024,
#         )
#         content = response.choices[0].message.content
#         print(content)
#         return content

#     async def a_generate(self, prompt: str) -> str:
#         response = await self.async_client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             max_tokens=1024,
#         )
#         content = response.choices[0].message.content
#         print(content)
#         return content

#     # ========================================================
#     # SHARED HELPERS
#     # ========================================================

#     def _build_system_prompt(self, schema) -> str:
#         schema_json = json.dumps(schema.model_json_schema(), indent=2)

#         return f"""You are an evaluation judge.

# Evaluate the input provided by the user and return ONLY one valid JSON object.

# Do NOT return markdown, ```json fences, explanations outside the JSON, or empty output.

# The JSON MUST follow this exact schema:

# {schema_json}

# Every required field MUST be present. Keep any "reason" field to ONE short
# sentence (under 25 words) — do not quote large blocks of source text.

# Return only the JSON object, nothing else."""

#     @staticmethod
#     def _clean_json_fences(content: str) -> str:
#         content = content.strip()
#         if content.startswith("```json"):
#             content = content[7:]
#         elif content.startswith("```"):
#             content = content[3:]
#         if content.endswith("```"):
#             content = content[:-3]
#         return content.strip()

#     @staticmethod
#     def _log_failure(tag: str, attempt: int, retries: int, response=None, note: str = "") -> None:
#         print(f"[OpenRouterJudge:{tag}] attempt {attempt + 1}/{retries} failed. {note}")
#         if response is not None:
#             try:
#                 print(response.model_dump())
#             except Exception as e:
#                 print(f"<could not dump response: {e}>")

#     # ========================================================
#     # SYNC STRUCTURED GENERATION (retry + auto max_tokens growth)
#     # ========================================================

#     def generate_with_schema(
#         self,
#         prompt: str,
#         schema: BaseModel,
#         retries: int = 4,
#         start_max_tokens: int = 1500,
#     ) -> BaseModel:

#         system_prompt = self._build_system_prompt(schema)
#         max_tokens = start_max_tokens
#         last_error = None

#         for attempt in range(retries):

#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0,
#                 max_tokens=max_tokens,
#                 response_format={"type": "json_object"},
#             )

#             choice = response.choices[0]
#             content = choice.message.content
#             finish_reason = choice.finish_reason

#             if not content:
#                 self._log_failure("sync", attempt, retries, response, "empty content")
#                 last_error = ValueError("OpenRouter returned empty response")
#                 time.sleep(2 ** attempt)
#                 continue

#             if finish_reason == "length":
#                 # Truncated: try again with a bigger budget, don't bother validating.
#                 self._log_failure("sync", attempt, retries, response,
#                                    f"truncated (finish_reason=length) at max_tokens={max_tokens}")
#                 max_tokens = min(max_tokens * 2, 8000)
#                 last_error = ValueError("Response truncated (finish_reason=length)")
#                 time.sleep(1)
#                 continue

#             content = self._clean_json_fences(content)

#             try:
#                 return schema.model_validate_json(content)
#             except Exception as e:
#                 self._log_failure("sync", attempt, retries, response, f"validation error: {e}")
#                 last_error = e
#                 time.sleep(2 ** attempt)
#                 continue

#         raise last_error

#     # ========================================================
#     # ASYNC STRUCTURED GENERATION (retry + auto max_tokens growth)
#     # ========================================================

#     async def a_generate_with_schema(
#         self,
#         prompt: str,
#         schema: BaseModel,
#         retries: int = 4,
#         start_max_tokens: int = 1500,
#     ) -> BaseModel:

#         system_prompt = self._build_system_prompt(schema)
#         max_tokens = start_max_tokens
#         last_error = None

#         for attempt in range(retries):

#             response = await self.async_client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0,
#                 max_tokens=max_tokens,
#                 response_format={"type": "json_object"},
#             )

#             choice = response.choices[0]
#             content = choice.message.content
#             finish_reason = choice.finish_reason

#             if not content:
#                 self._log_failure("async", attempt, retries, response, "empty content")
#                 last_error = ValueError("OpenRouter returned empty response")
#                 await asyncio.sleep(2 ** attempt)
#                 continue

#             if finish_reason == "length":
#                 self._log_failure("async", attempt, retries, response,
#                                    f"truncated (finish_reason=length) at max_tokens={max_tokens}")
#                 max_tokens = min(max_tokens * 2, 8000)
#                 last_error = ValueError("Response truncated (finish_reason=length)")
#                 await asyncio.sleep(1)
#                 continue

#             content = self._clean_json_fences(content)

#             try:
#                 return schema.model_validate_json(content)
#             except Exception as e:
#                 self._log_failure("async", attempt, retries, response, f"validation error: {e}")
#                 last_error = e
#                 await asyncio.sleep(2 ** attempt)
#                 continue

#         raise last_error

#     def get_model_name(self):
#         return self.model_name