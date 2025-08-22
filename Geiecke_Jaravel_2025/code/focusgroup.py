import hmac                 # Provides HMAC for message authentication
from openai import OpenAI   # OpenAI API client for making requests
import streamlit as st  # Imports Streamlit for building the web UI.
import time             # Imports time for timestamps and measuring durations.
import os               # Imports os for file and directory operations.
import requests         # Ollama API uses HTTP requests
import re    # Regular expressions for pattern matching
import json  # JSON parsing and serialization
import sys   # Ensures Streamlit can handle utf-8 unicode
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')



# =====================
# ===== DEBUGGING =====
# =====================
# Put this parameter on "ON" iff you want to see the prompts and the responses of the AIs in the Streamlit App.
# This might be helpful for writing the prompts.
# For instance, you can directly observe who the invisible moderator selects as next speaker.
DEBUGGING = False   # If you want to turn debugging off, set equal to "True". If you want to activate debugging, set equal to "False" (boolean variable).


# ===========================================
# ===== ACTIVATION OF HUMAN PARTICIPANT =====
# ===========================================
# If you want to do the focus group with a human participant set to "True", if you don't set to "False" (boolean variable).
HUMAN_ACTIVATION = False


# ==========================
# ====== CONFIG / SETUP ====
# ==========================
# AI model
MODEL = "gpt-5"  # or any other Open AI or Ollama ("gemma3") model 

# This can be run with many different OpenAI models which sometimes work better or worse (dependent for what purpose they were built).
# The best models are likely: "gpt-5-chat-latest" (better than "gpt-5" since this is the general model and not optimized for natural conversations) and "gpt-4o-2024-05-13" (for instance, used by Geiecke & Jaravel (2025)).
# The 'equivalent', smaller, more cost-efficient model is: "gpt-4o-mini-2024-07-18" (though reasoning is limited and therefore the transistions by the invisible moderator work rarely/almost never).
# If you want to use an Open AI key, you have to enter your key under 'secrets.toml' (folder: '.streamlit').

# We tried to do this with "gemma3" as Ollama model. But it seems like that the reasoning capacities of this model are not good enough.
# Especially the invisible moderator prompt often leads to not sensible results.

# Note that dependent on whether a reasoning model of OpenAI is used or not, the relevant token restriction is either '"max_output_tokens": MAX_OUTPUT_TOKENS' (no reasoning model) or '"max_completion_tokens": MAX_OUTPUT_TOKENS' (reasoning model)
# Therefore the correct token parameter has to be chosen based on the model (better check for each model that is not explicitly listed here what the correct paramter is).
# If the chosen token parameter is wrong you will get an error message.

# Maximum of output tokens for AI responses
MAX_OUTPUT_TOKENS = None   # Don't set this value too low. Otherwise you will receive an error message as the message can not be completed.
# Correct token parameter
def token_kwargs(model: str, max_out: int) -> dict:
    # GPT-5 family → max_output_tokens
    if model.startswith("gpt-5"):
        return {"max_completion_tokens": max_out}

    # Others
    else:
        return {"max_tokens": max_out}

token_params = token_kwargs(MODEL, MAX_OUTPUT_TOKENS)


# Temperature of the AI model (None is default value)
if MODEL == "gpt-5": # Model only has the default temperature
    TEMPERATURE_AI_PART = 1 # AI participants
    TEMPERATURE_INV_MOD = 1 # Invisible moderator
    TEMPERATURE_VIS_MOD = 1 # Visible moderator
else:
    TEMPERATURE_AI_PART = None # AI participants
    TEMPERATURE_INV_MOD = 0.1 # Invisible moderator
    TEMPERATURE_VIS_MOD = None # Visible moderator

# Display login screen with usernames and simple passwords for studies
LOGINS = False



# ===============================================
# ====== SET DIRECTORY TO STORE TRANSCRIPTS =====
# ===============================================
# Directories
TRANSCRIPTS_DIRECTORY = "../data_focusgroup/transcripts/"
TIMES_DIRECTORY = "../data_focusgroup/times/"
BACKUPS_DIRECTORY = "../data_focusgroup/backups/"

# Ensure required directories exist for storing transcripts and backups.
if not os.path.exists(TRANSCRIPTS_DIRECTORY):
    os.makedirs(TRANSCRIPTS_DIRECTORY)
if not os.path.exists(TIMES_DIRECTORY):
    os.makedirs(TIMES_DIRECTORY)
if not os.path.exists(BACKUPS_DIRECTORY):
    os.makedirs(BACKUPS_DIRECTORY)


# =======================================
# ====== TOPIC, GUIDELINES & CODES ======
# =======================================
# Define the focus group topic
TOPIC = "Reducing sedentary behavior while working from home — how can we promote healthier and more active remote work routines?"

OVERALL_DURATION = "60 minutes"

FOCUS_GROUP_OUTLINE = f"""
AI-Led Focus Group Outline — Employees Case

Part 0 – Welcome, Introduction, Housekeeping & Orientation
- Moderator welcomes participants, introduces him/herself, discussion rules (housekeeping), the topic and its motivation, 
  gives a brief overview about the discussion guide and discussion duration, and invites participants to introduce themselves and how they relate to the topic.
- Participants introduce themselves and, if at all, comment briefly on how they feel related to the topic.

Part 1 - Knowledge & Awareness about Sitting Behavior
- Moderator shares information on risks of too much sitting and statistics on the effect of working from home on humans' sitting behavior with participants:
    - High levels of sitting are associated with:
        - Increased risk of all-cause cardiovascular disease and cancer mortatility
        - Increased likelihood of developing cardiovascular disease, some cancers, or type 2 diabetes
        - Increased risk of anxiety, depression, and sleep disorders
        - Lower Levels of emotional wellbeing
        --> These risks can be reduced by moving throughout the day.
    - Working at home changes the sitting behavior of 58%-89% of all office workers (dependent on the study)
- Moderator asks participants about their knowledge, awareness, and opinion on this information.
- Participants share opinions/thoughts/experiences and may discuss.(Moderator non-actively steers the discussion and may ask clarifying questions, or try to deepen the discussion).

Part 2 - Current Sitting Behavior of Participants & Influences
- Moderator asks participants to share their sitting behavior, whether they sit more or less than before working in the office, and what influences this.
- Participants share opinions/thoughts/experiences and may discuss.(Moderator non-actively steers the discussion and may ask clarifying questions, or try to deepen the discussion).

Part 3 - Ideas/Solutions to Improve Sitting Behavior
- Moderator asks participants for ideas to break up sitting behavior:
    - Is there anything you can think of that would be a good idea for breaking up sitting - even if you haven't already tried it out yourself? [moderator shall mention this question in the question that transition to this part of the topic guide]
    - There are some ideas that have been identified to work in a regular office environment to improve sitting behavior (education about sitting behavior and the health consequences, regular prompts, personal/individual feedback on sitting behavior).
      How do participants think they would work at home? [Moderator asks this as a Deepening question but does not use this to transition from the last part to this part of the discussion guide.]
- Participants share opinions/thoughts/experiences and may discuss.(Moderator non-actively steers the discussion and may ask clarifying questions, or try to deepen the discussion).

Part 4 - Wrap-Up, Feedback & Goodbye
- Moderator recaps/summarises focus group discussion in own words, thanks for participation, and asks for feedback
- If all participants, who wanted, provided feedback, the moderator concludes the discussion.
"""

# Define closing messages for different codes
# Codes
CODES = f"""Codes:  

Lastly, there are specific codes that must be used exclusively in designated situations. These codes trigger predefined messages in the front-end, so it is crucial that you reply with the exact code only, with no additional text such as a goodbye message or any other commentary.

Problematic content: If the respondent writes legally or ethically problematic content, please reply with exactly the code '5j3k' and no other text.

End of the focus group: When you have asked all questions from the Focus Group Outline, or when the respondent does not want to continue the focus group, please reply with exactly the code 'x7y8' and no other text.
"""

# Pre-written closing messages for codes
CLOSING_MESSAGES = {}
CLOSING_MESSAGES["5j3k"] = "Thank you for participating, the focus group concludes here."
CLOSING_MESSAGES["x7y8"] = (
    "So we are done! This is the end of the focus group! Thank you very much for your participation!"
)

GENERAL_INSTRUCTIONS = f"""
- Guide the discussion in a non-directive, neutral way. Encourage participants to share their perspectives freely without leading or suggesting answers.  
- Ask open-ended follow-up questions to clarify points, deepen understanding, and encourage examples. For example:  
  'Can you tell us more about the last time you experienced that?',  
  'What has that been like for you?',  
  'Why do you think this matters?',  
  or 'Could you share a specific example?'  
  Tailor your questions naturally to the context and avoid repeating these examples verbatim.  
- Encourage participants to describe specific events, situations, people, practices, or experiences relevant to the topic. Avoid broad generalizations.  
- Show cognitive empathy by exploring how and why participants hold their views. Ask about the origins, coherence, and implications of their beliefs.  
- Foster an inclusive environment where diverse views are welcomed. Avoid questions that assume particular views or provoke defensiveness.  
- Maintain turn-taking: do not ask multiple questions at once or interrupt participants.  
- Keep the discussion focused on the purpose of the focus group. If the conversation drifts, gently redirect back to the topic.  
"""


#========================================================
# ====== DEFINE AI AGENT PERSONAS FOR PARTICIPANTS ======
#========================================================
# Define AI agent personas for each participant in the focus group.
# The display role refers to the role of the AI agents in the Streamlit chat and is set to assistant.

# The code is written in a way that allows for easy addition of new participants by simply adding a new dictionary entry to the `participants` dictionary.

participants = {

    "Participant 1": {
        "display_role": "user",
        "name": "Amelia",
        "avatar": "👩‍💼",
        "prompt": f"""
You are Amelia (30, UK), a Public Involvement Coordinator in health research.
You’re thoughtful, articulate, and often draw on real-life adjustments you’ve made when working from home.

You are participating in an online focus group on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Here is the ongoing conversation so far:
{{chat_history}}

Respond as Amelia would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
                """
    },

    "Participant 2": {
        "display_role": "user",
        "name": "Noah",
        "avatar": "👨‍🔬",
        "prompt": f"""
You are Noah (26, UK), a Research Fellow in public health.
You’re curious, analytical, and sometimes surprised by new evidence.

You are participating in an online focus group on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Here is the ongoing conversation so far:
{{chat_history}}

Respond as Noah would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
                """
    },

    "Participant 3": {
        "display_role": "user",
        "name": "Mia",
        "avatar": "👩‍🍳",
        "prompt": f"""
You are Mia (38, UK), a General Practitioner working from a small home space.
You speak plainly, often noting the practical constraints of your environment.

You are participating in an online focus group on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Here is the ongoing conversation so far:
{{chat_history}}

Respond as Mia would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
                """
    },

    "Participant 4": {
        "display_role": "user",
        "name": "Oliver",
        "avatar": "👨‍🏫",
        "prompt": f"""
You are Oliver (36, UK), a university lecturer in social sciences.
You’re reflective, link personal habits to wider patterns, and sometimes admit to shifting your views.

You are participating in an online focus group on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Here is the ongoing conversation so far:
{{chat_history}}

Respond as Oliver would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
                """
    },

    "Participant 5": {
        "display_role": "user",
        "name": "Harper",
        "avatar": "👩‍🔧",
        "prompt": f"""
You are Harper (37, UK), a Facilities Officer who values flexibility and work-life balance.
You share concrete examples from your workplace policies and personal adjustments.

You are participating in an online focus group on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Here is the ongoing conversation so far:
{{chat_history}}

Respond as Harper would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
                """
    },

    "Participant 6": {
    "display_role": "user",
    "name": "Lucy",
    "avatar": "👩‍🔬",
    "prompt": f"""
You are Lucy (30, UK), a Public Health Intelligence Advisor.
You’re observant, often noticing small everyday movements lost when working from home, like walking to meetings or getting coffee.

You are participating in an online focus group on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Here is the ongoing conversation so far:
{{chat_history}}

Respond as Lucy would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
                """
    }

}

# Only add the AI-only participant if human activation is not ON
if HUMAN_ACTIVATION != True:
    participants["Participant 7"] = {
        "display_role": "user",
        "name": "Ethan",
        "avatar": "🤖",
        "prompt": f"""
You are Ethan, a 32-year-old computer scientist from Germany working at SAP. 
You are attentive and curious, often noticing subtle details, like someone stretching at their desk—things others might not pick up on.

You’re participating in an online focus group about '{TOPIC}', motivated by the experience of finding new ways of working due to COVID lockdowns.

Here's the discussion so far:
{{chat_history}}

Respond as Ethan would, contributing naturally to the current conversation. Generate your message with an individual, natural sentence structure that is distinct from other agents using this prompt structure. Create only one message per turn, without generating dialogue for other participants. 
Speak in the first person, react to the latest comment, and avoid putting statements in parentheses. 
Do not begin your introduction with the same greetings structure as the others (e.g., “Hi/Hey everyone...”) or your reply with repetitive sentence starters (like always beginning with “I...”). 
Also, refrain from repeating your or others’ names or repeating previous messages verbatim.
Vary the length and style of your contributions so your responses feel authentic and unique compared to other agents. 
Short responses may be appropriate for simple agreement, while deeper topics can justify longer—yet concise—replies. 
Most comments should be a maximum of 3–4 concise lines. Use diverse sentence structures and phrasing to further ensure your responses are distinct and engaging.
       """
    }

# Map participant names to their dictionary keys
name_to_participant_key = {p_data["name"]: key for key, p_data in participants.items()}


# ======================================================
# ====== DEFINE SUMMARY VARIABLES OF PARTICIPANTS ======
# ======================================================
# Define lists of participant names for different contexts.
ALL_VISIBLE_PARTICIPANTS = (
    [p_data["name"] for p_data in participants.values()] +
    (["You"] if HUMAN_ACTIVATION == True else []) +
    ["Moderator"]
)

VISIBLE_PARTICIPANTS = (
    [p_data["name"] for p_data in participants.values()] +
    (["You"] if HUMAN_ACTIVATION == True else [])
)
AI_PARTICIPANTS = [p_data["name"] for p_data in participants.values()]



    
# ================================
# ====== MODERATOR AI AGENT ======
# ================================

#===== Visible Moderator =====
# Define name of moderator
NAME_MODERATOR = "Anne"  # Similarly, also the job and occupation, etc. of the moderator could be defined.

# Define the visible moderator's persona and prompts
visible_moderator = {
    "display_role": "assistant",  # The visible moderator acts as an assistant in the Streamlit chat.
    "name": "Moderator ({NAME_MODERATOR})",
    "avatar": "🗨️",
    "prompt_introduction": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion rules:
- There are no right and wrong answers. All opinions are valuable.
- Everything that is said, is treated confidentially. Publications won't allow to identify a subject.
- Please do not share discussion outside this room.
- Please be repectful of everyone's opinions.

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Task: Welcome the participants. Introduce yourself (you are a researcher on the topic and regularly conduct focus groups), the topic and the topic's motivation.
Tell them that you are, in particular, interested in understanding the challenges they face and indetifying opportunities to break up sitting while working from home.
Briefly mention the discussion outline without going into detail. Do NOT mention how much time is allocated to which part of the conversation!
In addition, briefly mention how long round about to talk about the different parts of the discussion guide. Share the discussion rules with the participants.
Invite each participant to introduce themselves with their name, age, occupation, and briefly comment on how they relate to the topic.
Avoid enumerations when outlining the course of the focus group.

Example: "Hello, my name is {NAME_MODERATOR}, I am 35 years old and I research working from home..."

Only create one message how you as the moderator would start the focus group. Do NOT hallucinate a whole focus group, i.e. answers of participants.

Always speak in the first person, keep your message concise (as short as pissible but as long as necessary), answer not in parantheses (""),
and do not start with your own or another participant’s name.

Follow these general moderation instructions: 
{GENERAL_INSTRUCTIONS}.
                            """,

    "prompt_transition": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Chat history:
{{chat_history}}

Task: Smoothly - look at the past conversation (chat history) - move from the current completed part of the topic guideline (Part {{transition_count}}) to the next part of the focus group discussion.  
- It can be helpful to summarize key points from the previous section.  
- Introduce the next section naturally (do not read the outline verbatim).  
- Prompt participants to share their thoughts (optionally ask a question to start the new section).
- Follow the descriptions in Part {{transition_count}} of the Discussion outline what the moderator has to do in this section.

Only create one message how you as the moderator of the focus group would transition to the next part of the guideline. Do NOT hallucinate a whole focus group, i.e. answers of participants.

Always speak in the first person, keep your message concise (as short as pissible but as long as necessary), answer not in parantheses (""), react to the most recent discussion, answer not in parantheses (""),
and do not repeat earlier messages verbatim or start with your own or another participant’s name.

Follow these general moderation instructions: 
{GENERAL_INSTRUCTIONS}.
                            """,

    "prompt_topic_deepening": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Chat history:
{{chat_history}}

Task: Right now, the focus group is talking about Part {{transition_count}}. Build on interesting but underexplored points (most likely from the previous message, or the most recent messages in the chat history).  
Ask a follow-up question that encourages deeper insights and keeps the discussion on-topic. Also look whether all of the main question you should ask participants in this part from the disscussion outline have been covered.
Otherwise, use these questions (not verbatim) to deepen the conversation.

Only create one message how you as the moderator of the focus group would motivate participants to deepen their conversation on a specific thought or topic. Do NOT hallucinate a whole focus group, i.e. answers of participants.

Always speak in the first person, keep it to at most 3–4 concise lines, react to the most recent discussion, answer not in parantheses (""),
and do not repeat earlier messages verbatim or start with your own or another participant’s name.

Follow these general moderation instructions:
{GENERAL_INSTRUCTIONS}.
                            """,

    "prompt_claryfying_question": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Chat history:
{{chat_history}}

Task: A participant has shared something interesting but unclear (most likely previous message(s) of the chat history).  
Ask a short clarifying question to help them explain their point and keep the discussion relevant.
For your orientation, the focus group discussion is in part {{transition_count}} of the discussion outline right now.

Only create one message how you as the moderator of the focus group would clarify unclear points raised by a participant in the previous message. Do NOT hallucinate a whole focus group, i.e. answers of participants.

Always speak in the first person, keep it to at most 3–4 concise lines, react to the most recent discussion, answer not in parantheses (""),
and do not repeat earlier messages verbatim or start with your own or another participant’s name.

Follow these general moderation instructions: 
{GENERAL_INSTRUCTIONS}.
                            """,

    "prompt_closing_statement": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Chat history:
{{chat_history}}

Task: The discussion is complete (either all parts are covered or participants have nothing more to add).  
Give a short closing statement:

{CODES}

Follow these general moderation instructions: 
{GENERAL_INSTRUCTIONS}.
                            """,

    "prompt_topical_continuity": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Chat history:
{{chat_history}}

Task: The conversation has drifted off-topic or stalled (look at the previous message(s) of the chat history).  
Gently guide participants back to the focus group topic and encourage further input.

Only create one message how you as the moderator of the focus group would steer participants back on topic. Do NOT hallucinate a whole focus group, i.e. answers of participants.

Always speak in the first person, keep it to at most 3–4 concise lines, react to the most recent discussion, answer not in parantheses (""),
and do not repeat earlier messages verbatim or start with your own or another participant’s name.

Follow these general moderation instructions: 
{GENERAL_INSTRUCTIONS}.
                            """,

    "prompt_general": f"""
You are the Moderator of an online focus group. Your name is {NAME_MODERATOR}.

The focus group is on '{TOPIC}' and motivated by COVID lockdowns having led to new ways of working. 

Discussion outline: 
{FOCUS_GROUP_OUTLINE}.

Chat history:
{{chat_history}}

Task: Continue the discussion naturally, keeping it relevant to the topic and aligned with the discussion guide. For your orientation, the conversation is in Part {{transition_count}} of the topic guide right now. 
Ask questions or make comments that sustain engagement.

Only create one message how you as the moderator of the focus group would proceed in ensuring a natural conversation/discussion flow of the focus group. Do NOT hallucinate a whole focus group, i.e. answers of participants.

Always speak in the first person, keep it to at most 3–4 concise lines, react to the most recent discussion, answer not in parantheses (""),
and do not repeat earlier messages verbatim or start with your own or another participant’s name.

Follow these general moderation instructions: 
{GENERAL_INSTRUCTIONS}.
                            """
}

#===== Invisible Moderator =====
# Define the invisible moderator's persona and prompt.
invisible_moderator = {
    "display_role": "system",  # The invisible moderator acts as a system in the Streamlit chat.
    "name": "Invisible Moderator",
    "avatar": "👤",
    "prompt": f"""
You are the Invisible Moderator of an online focus group, silently observing and selecting the next speaker, or transitioning to the next discussion part as appropriate. Base decisions on elapsed time, discussion rules, and participation.

## Context
- Topic: {TOPIC}

- Discussion Guide/Outline: 
  {FOCUS_GROUP_OUTLINE}
  
- Participants: {ALL_VISIBLE_PARTICIPANTS}

- Note: In case, you read 'You' as a participant's name this refers to a participant, not yourself.

- The participants were selected since they are employees and have experienced working from home in the past.

- The focus group is motivated by the COVID lockdown leading to new ways of working.

## Conversation State

Chat history:

*****************************************************************************************************************************************************
{{chat_history}}
*****************************************************************************************************************************************************

Speaking time by participant (minutes): {{speaking_time}}

Elapsed time: {{time_spent}} minutes

Current discussion part of discussion guide: {{transition_count}} 


## Guidelines for choosing the next speaker:

Follow these priorities and considerations, but allow some flexibility to ensure natural, engaging conversation flow:

1. If the chat history is empty or no moderator message has appeared yet:

   Return: `Moderator ({NAME_MODERATOR})_prompt_introduction`

   Note that this is the only case where to return this prompt.

2. Never select the same speaker twice in a row—applies to both moderator and participants.

3. Prefer participants who have spoken less or less recently. Aim for even speaking time; minor 2–3 min differences are acceptable. Moderator only speaks when needed.

4. Prioritize topical continuity; select the participant best positioned to address the most recent point.

5. If the conversation has drifted off-topic or needs refocusing, the moderator can gently steer it back:

   Return: `Moderator ({NAME_MODERATOR})_prompt_topical_continuity`

6. If a participant's last input is unclear:

   Return: `Moderator ({NAME_MODERATOR})_prompt_claryfying_question`

7. If an idea requires deepening:

   Return: `Moderator ({NAME_MODERATOR})_prompt_topic_deepening`

8. If a participant was just directly addressed, pick them next.

9. If discussion is ending, time is almost up, interest is waning, and especially if participants write legally or ethically problematic content: 

    Return: `Moderator ({NAME_MODERATOR})_prompt_closing_statement`

10. If moderator is next for natural flow and no above rule applies:

    Return: `Moderator ({NAME_MODERATOR})_prompt_general`


## Output options (choose exactly one):

- For participants: use their exact names from {ALL_VISIBLE_PARTICIPANTS} (Note: 'You' refers to a participant, not yourself.)

- For the moderator, choose one of:

  - Moderator ({NAME_MODERATOR})_prompt_introduction

  - Moderator ({NAME_MODERATOR})_prompt_transition

  - Moderator ({NAME_MODERATOR})_prompt_topic_deepening

  - Moderator ({NAME_MODERATOR})_prompt_claryfying_question

  - Moderator ({NAME_MODERATOR})_prompt_closing_statement

  - Moderator ({NAME_MODERATOR})_prompt_topical_continuity

  - Moderator ({NAME_MODERATOR})_prompt_general


## Important Instructions

1. **Output format**  
   - Output only the chosen code string.  
   - Do not add explanations or extra text.  

2. **Decision inputs**  
   - Base your decision on:
     - The chat history (between the *** lines)
     - The participants’ speaking times
     - The elapsed time compared to the discussion guide schedule

3. **Introduction round**  
   - After the moderator’s first introduction message, ensure that each participant speaks once.
   - Go around systematically until all participants have spoken once.
   - Afterwards transition to Part 1 of the discussion guide. Return: `Moderator ({NAME_MODERATOR})_prompt_transition`.

4. **Smooth conversation flow**  
   - Make the dialogue feel natural, coherent, and balanced.  
   - Prefer participants who have spoken less or less recently.  
   - Minor differences of 2–3 minutes in speaking time are acceptable.  

5. **Speaker restriction rule**  
   - **Never select the same speaker twice in a row.**  
   - If the last speaker was participant X, X cannot be selected next.  
   - If the last speaker was the moderator, you cannot select a moderator prompt next.  
   - This rule is absolute and must always be enforced.  

6. **Timing and transitions**  
   - Always keep track of elapsed time: {{time_spent}} minutes.  
   - Recall the focus group is in part {{transition_count}} right now.
   - Each part of the discussion guide has a target transition time. ENSURE the TRANSITIONS ARE MADE AT THE RIGHT TIME and the FOCUS GROUP ENDS EXACTLY AFTER 60 MINUTES:
     - Part 0 (intro) → immediately transition to Part 1 if everyone has responded to the moderator's first message by introducing themselves once. Don't allow that participants react with a second message to othe rparticipants' comments in Part 0 (Intro). 
     - Part 1 → transition to Part 2 when the **elapsed time** amounts to **21-23 minutes**.
     - Part 2 → transition to Part 3 when the **elapsed time** amounts to **38-40 minutes**.
     - Part 3 → transition to Part 4 when the **elapsed time** amounts to **55-57 minutes**.
     - Part 4 (final) → close the focus group session when the **elapsed time** amounts to **60 minutes**.
   - Don't finish early, i.e., several minutes before the elapsed time amounts to 60 minutes, unless participants actively say that they don't know what they could contribute anymore. 
   -  Only transition when the elapsed time window for this part is reached (e.g., Part 1 ends at 20–25 minutes) or you really have the feeling that no partcipant could add anything new to the part (in particular, if they say that they do not know what to add anymore), you must transition with:  
     `Moderator ({NAME_MODERATOR})_prompt_transition`  
   - At the end (Part 4), close with:  
     `Moderator ({NAME_MODERATOR})_prompt_closing_statement`  
   - Do not end earlier unless participants misbehave.

Especially follow 5. **Speaker restriction rule** and 6. **Timing and transitions**  to decide when to select the moderator and not a participant!
Most importantly, Participants may (and should) speak multiple times in each part, except in the introduction round (Part 0). Do not transition simply because everyone has spoken once.
The focus group must last exactly 60 minutes unless participants explicitly say they have nothing more to add or misbehave. Do not end early, even if all discussion guide bullet points have been covered. 
In case "You" is a participant, you must select "You" to give him the opprtunity to give feedback before you conclude the discussion. You do not necessarily have to give this option to all other participants.
"""
}


# ==============================================
# ====== DEFINE HUMAN PERSONA PLACEHOLDER ======
# ==============================================
# The human participant is represented by a simple dictionary with display role, name, and avatar.
# The display role is set to "User" to indicate that this is the role of the human participant in the Streamlit chat.
if HUMAN_ACTIVATION == True:
    human = {"display_role": "User", "name": "You", "avatar": "🧑‍💻"}


# ======================================================
# ====== DEFINE SUMMARY VARIABLES OF PARTICIPANTS ======
# ======================================================

# A streamlit app requires roles for chat messages.
# Usually, these roles are 'user' for the respondent and 'assistant' for the AI interviewer, and 'system' for the invisible moderator.
# However, we also have a 'Moderator' role for the visible moderator and several AI participants.
# Therefore, we define a dictionary that maps the roles to their display names and avatars.

# Define useful dictionaries
# Build ROLES dictionary dynamically
ROLES = {
    **(
        {"You": human} if HUMAN_ACTIVATION == True else {}
    ),
    **{p_data["name"]: p_data for p_data in participants.values()},
    "Invisible Moderator": invisible_moderator,
    "Moderator": visible_moderator
}

# ============================================================================================================================
# ====== DEFINITION OF HELPER FUNCTIONS: LOGIN, CHECK WHETHER FOCUS GROUP HAS ALREADY BEEN COMPLETED, SAVING TRANSCRIPTS =====
# ============================================================================================================================
# Password screen for dashboard (note: only very basic authentication!)
# Based on https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
def check_password():
    """Returns 'True' if the user has entered a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether username and password entered by the user are correct."""
        if st.session_state.username in st.secrets.passwords and hmac.compare_digest(
            st.session_state.password,
            st.secrets.passwords[st.session_state.username],
        ):
            st.session_state.password_correct = True

        else:
            st.session_state.password_correct = False

        del st.session_state.password  # don't store password in session state

    # Return True, username if password was already entered correctly before
    if st.session_state.get("password_correct", False):
        return True, st.session_state.username

    # Otherwise show login screen
    login_form()
    if "password_correct" in st.session_state:
        st.error("User or password incorrect")
    return False, st.session_state.username

# Check if interview transcript/time file exists which signals that interview was completed.
def check_if_focusgroup_completed(directory, username):
    """
    Check if any focus group transcript file exists that starts with the username.
    Special case: always return False for username == "testaccount".
    """
    if username == "testaccount":
        return False
    else:
        for fname in os.listdir(directory):
            if fname.startswith(username):
                return True
        return False

# Save focusgroup data (transcript and time) to disk.
# This function writes the chat transcript and timing information to specified directories.
def save_focusgroup_data(
    transcript_path,
    transcript_full_path,
    time_path
):
    """Save focus group data to fixed paths (overwrites existing files)."""

    # --- Chat transcript ---
    with open(transcript_path, "w", encoding="utf-8") as t:
        for message in st.session_state.messages:
            t.write(f"{message['role']}: {message['content']}\n")

    # --- Full transcript with invisible moderator ---
    with open(transcript_full_path, "w", encoding="utf-8") as t:
        for message in st.session_state.messages_full:
            t.write(f"{message['role']}: {message['content']}\n")

    # --- Timing information ---
    with open(time_path, "w", encoding="utf-8") as d:
        duration = (time.time() - st.session_state.start_time) / 60
        d.write(
            f"Start time (UTC): {time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(st.session_state.start_time))}\n"
            f"Interview duration (minutes): {duration:.2f}"
        )

# ===========================
# ====== SET TITLE PAGE =====
# ===========================
# Set page title and icon for the Streamlit app.
st.set_page_config(page_title="AI-led Focus Group", page_icon="🎤")

# Main title
st.title("🎤 AI-led Focus Group — Working from Home")

# Short info text in an info box
if HUMAN_ACTIVATION == True:
    st.info(
    "💡 **Welcome to the online focus group!**\n\n"
    "You are one of several participants in this discussion.\n\n"
    "You can send a message **only** when the system selects you to speak — "
    "a text box will appear when it’s your turn.\n\n"
    "Please read what others have shared, and feel free to respond to their points "
    "or introduce your own ideas.\n\n" 
    "To make the start of the focus group easier, please introduce yourself with your name and occupation in your first message.\n\n" 
    "Please do not share any legally or ethically problematic content. Otherwise the focus group will end immediately. "
    "You can always end the conversation by clicking on the 'Quit' button."
)


# ===========================
# ====== SET LOGIN PAGE =====
# ===========================
# Check if login functionality is enabled.
if LOGINS:
    # Display login form and check password.
    pwd_correct, username = check_password()  # Returns True and username if login is successful.
    if not pwd_correct:                       # If password is incorrect, stop the app.
        st.stop()
    else:
        st.session_state.username = username  # Store username in session state.
else:
    st.session_state.username = "testaccount" # Use default username if logins are disabled.


# ==============================
# ====== SET SESSION STATE =====
# ==============================

# Initialize session state variables for the interview/focus group process.
# Session state is used to maintain the state of the application across reruns.

# Initialize interview activity state in session.
if "interview_active" not in st.session_state:
    st.session_state.interview_active = True  # Interview is active by default.

# Initialize chat message history in session (ensures chat histories exist)
if "messages" not in st.session_state:
    st.session_state.messages = []  # main history (no Invisible Moderator)

if "messages_full" not in st.session_state:
    st.session_state.messages_full = []  # full history (with Invisible Moderator)

# Store interview start time and formatted timestamp in session.
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time() # Record current time as start.
    st.session_state.start_time_file_names = time.strftime(
        "%Y_%m_%d_%H_%M_%S", time.localtime(st.session_state.start_time)
    )  # Format start time for filenames.


# =========================
# ====== LOAD MODELS ======
# =========================
# Load API library depending on model type.
if "gpt" in MODEL.lower():  # Checks if the model name contains 'gpt' (case-insensitive).
    api = "openai"                 # Sets API type to OpenAI.
    client = OpenAI(api_key=st.secrets["API_KEY_OPENAI"])

# --- BEGIN: Added Ollama option for gemma3 model ---

elif "gemma3" in MODEL.lower():
    api = "ollama"            # Sets API type to Ollama for local models.

# --- END: Added Ollama option for gemma3 model --

else:
    raise ValueError(                  # Raises error if model type is not recognized.
        "Model does not contain 'gpt' or 'gemma3'; unable to determine API."
    )


# =================================================================
# ====== CHECK WHETHER FOCUS GROUP HAS ALREADY BEEEN COMPLETED =====
# =================================================================
# Check if the focus group has already been completed for this user.
focusgroup_previously_completed = check_if_focusgroup_completed(
    TIMES_DIRECTORY, st.session_state.username
)

# If focus group has been completed and no messages are present, mark as inactive.
if focusgroup_previously_completed and not st.session_state.messages:
    st.session_state.interview_active = False
    completed_message = "Focus group already completed."
    st.markdown(completed_message)  # Display completion message.


# ============================
# ====== ADD QUIT BUTTON =====
# ============================
# Add a 'Quit' button to the dashboard using Streamlit columns.
col1, col2 = st.columns([0.85, 0.15])  # Layout: main area and small button area.
with col2:
    # If interview is active and user clicks 'Quit', end interview.
    if st.session_state.interview_active and st.button(
        "Quit", help="End the focus group."
    ):
        st.session_state.interview_active = False  # Mark interview as inactive.
        quit_message = "You have cancelled the focus group."
        st.session_state.messages.append({"role": "assistant", "content": quit_message})  # Log quit message.
        save_focusgroup_data(
            st.session_state.username,
            TRANSCRIPTS_DIRECTORY,
            TIMES_DIRECTORY,
        )  # Save transcript and timing data.


# =============================================
# ====== WEBPAGE UPDATING AFTER EVERY RUN =====
# =============================================
# On rerun, display previous conversation (excluding system prompt/first message).
for message in st.session_state.messages[1:]:

    names = list(ROLES.keys())

    for name in names:
    # Skip messages with closing codes
        if name == message["role"] and not any(code in message["content"] for code in CLOSING_MESSAGES.keys()):

            # Display message with the role's display_role and avatar
            with st.chat_message(message["role"], avatar=ROLES[name]["avatar"]):
                st.markdown(f"**{message['role']}:** {message['content']}")


# ============================
# ===== FOCUS GROUP LOOP =====
# ============================

# Define file names for backup files
if "backups_paths" not in st.session_state:

    if HUMAN_ACTIVATION == True:
        base = f"{st.session_state.username}_{MODEL}_HUMAN_{st.session_state.start_time_file_names}"
    else: 
        base = f"{st.session_state.username}_{MODEL}_NO_HUMAN_{st.session_state.start_time_file_names}"

    st.session_state.backups_paths = {
        "transcript": os.path.join(BACKUPS_DIRECTORY, f"{base}.txt"),
        "transcript_full": os.path.join(BACKUPS_DIRECTORY, f"{base}_with_inv_mod.txt"),
        "time": os.path.join(BACKUPS_DIRECTORY, f"{base}_time.txt"),
    }

# Define file names for final transcripts
if "transcript_paths" not in st.session_state:

    if HUMAN_ACTIVATION == True:
        base = f"{st.session_state.username}_{MODEL}_HUMAN_{st.session_state.start_time_file_names}"
    else: 
        base = f"{st.session_state.username}_{MODEL}_NO_HUMAN_{st.session_state.start_time_file_names}"

    st.session_state.transcript_paths = {
        "transcript": os.path.join(TRANSCRIPTS_DIRECTORY, f"{base}.txt"),
        "transcript_full": os.path.join(TRANSCRIPTS_DIRECTORY, f"{base}_with_inv_mod.txt"),
        "time": os.path.join(TIMES_DIRECTORY, f"{base}_time.txt"),
    }


if st.session_state.interview_active:

    # Take time to build the website. Don't start immediately with the website:
    time.sleep(3)

    while st.session_state.interview_active:

        # Add break to keep the token rate per minute low enough
        time.sleep(6)

        # === 1. Build current chat context ===

        # Updated conversation history
        chat_history = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in st.session_state.messages
            if ROLES.get(m["role"], {}).get("display_role") != "system"
        )

        # Cummulated number of 'spoken' words
        total_words = sum(
            len(re.findall(r"\w+", m["content"]))
            for m in st.session_state.messages
            if ROLES.get(m["role"], {}).get("display_role") != "system"
        )

        # Cumulated elapsed time
        time_spent = round(total_words / 100.0, 1)

        # Function to count words for every participant and moderator
        def word_count(text):
            if not isinstance(text, str):
                return 0
            return len(re.findall(r"\w+", text))
        
        # Messages of every indvidual + function to calculate speaking time of every participant and moderator
        messages = st.session_state.get("messages", [])

        def minutes_spoken(name):
            words = sum(word_count(m.get("content", "")) for m in messages
                        if m.get("role") == name)
            return round(words / 100.0, 1)  # adjust divisor to your wpm heuristic

        speaking_time_str = ", ".join(f"{name} ({minutes_spoken(name)} min)"
                              for name in ALL_VISIBLE_PARTICIPANTS)
        
        # Count transitions so that moderator knows which part of the discussion guide to deal with
        transition_count = sum(
                1 for msg in st.session_state.get("messages_full", []) 
                if msg.get("content") == f"Moderator ({NAME_MODERATOR})_prompt_transition"
                )
        
        
        # === 2. Fill invisible moderator prompt ===

        inv_filled = (
            invisible_moderator["prompt"]
            .replace("{chat_history}", chat_history)
            .replace("{time_spent}", str(time_spent))
            .replace("{speaking_time}", speaking_time_str)
            .replace("{transition_count}", str(transition_count))
        )

        # Debug: show invisible moderator prompt
        if DEBUGGING == True:
            st.subheader("🔍 Invisible Moderator Prompt")
            st.code(inv_filled)


        # === 3. Call invisible moderator model ===

        if api == "openai":
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": inv_filled}],
                temperature=TEMPERATURE_INV_MOD,
                **token_params
            )
            next_speaker = response.choices[0].message.content.strip()

        elif api == "ollama":
            response = requests.post(
                url="http://localhost:11434/api/chat",
                json={
                    "model": MODEL,
                    "messages": [{"role": "system", "content": inv_filled}],
                    "stream": False, # Note that there is no real option to use streaming here. If streaming is set to "True" the response is returned in chunks which the following code is not able to return. If you want to use streaming you have to adjust the code.
                    "temperature": TEMPERATURE_INV_MOD,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                }
            )
            data = response.json()
            next_speaker = data["message"]["content"].strip()

        else:
            st.error(f"Unknown API type: {api}")
            st.stop()


        # Debug: show raw invisible moderator output
        if DEBUGGING == True:
            st.subheader("📤 Invisible Moderator Output")
            st.code(response)
            st.code(next_speaker)


        # Store invisible moderator message in full log
        st.session_state.messages_full.append({"role": "Invisible Moderator", "content": next_speaker})

        
        # === 4. Handle turns ===

        # --- Human turn ---
        if HUMAN_ACTIVATION == True and next_speaker == "You":
            user_msg = st.chat_input("Your message here")
            if not user_msg:
                st.stop()
            st.session_state.messages.append({"role": "You", "content": user_msg})
            st.session_state.messages_full.append({"role": "You", "content": user_msg})
            with st.chat_message(ROLES["You"]["display_role"], avatar=ROLES["You"]["avatar"]):
                st.markdown(f"**{next_speaker}:** {user_msg}")


        # --- AI participant turn ---
        elif next_speaker in name_to_participant_key:
        # Update prompt of AI participant that is next_speaker
            pdata_key = name_to_participant_key[next_speaker]
            pdata = participants[pdata_key]
            prompt_text = (
                pdata["prompt"]
                .replace("{chat_history}", chat_history)
                .replace("{time_spent}", str(time_spent))
                .replace("{speaking_time}", speaking_time_str)
                .replace("{transition_count}", str(transition_count))
            )

            # Debug: show ai participant prompt
            if DEBUGGING == True:
                st.subheader("🔍 AI Participant Prompt")
                st.code(prompt_text)

            # Call AI model
            if api == "openai":
                response_ai = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=TEMPERATURE_AI_PART,
                    **token_params
                )
                ai_msg = response_ai.choices[0].message.content.strip()

            elif api == "ollama":
                response_ai = requests.post(
                    url="http://localhost:11434/api/chat",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "stream": False, # Note that there is no real option to use streaming here. If streaming is set to "True" the response is returned in chunks which the following code is not able to return. If you want to use streaming you have to adjust the code.
                        "temperature": TEMPERATURE_AI_PART,
                        "max_tokens": MAX_OUTPUT_TOKENS,
                    }
                )
                data_ai = response_ai.json()
                ai_msg = data_ai["message"]["content"].strip()

            # Debug: show AI participant output
            if DEBUGGING == True:
                st.subheader("📤 AI Participant Output")
                st.code(response_ai)
                st.code(ai_msg)

            # Show AI model output in App as chat message
            with st.chat_message(ROLES[next_speaker]["display_role"], avatar=ROLES[next_speaker]["avatar"]):
                st.markdown(f"**{next_speaker}:** {ai_msg}")

            # Store AI participant's message in log files
            st.session_state.messages.append({"role": next_speaker, "content": ai_msg})
            st.session_state.messages_full.append({"role": next_speaker, "content": ai_msg})


        # --- Visible moderator turn ---
        elif next_speaker.startswith(f"Moderator ({NAME_MODERATOR})"):

            # Choose the right prompt
            suffix = next_speaker.split("_", 1)[1]
            prompt_text = visible_moderator.get(suffix)
            if not prompt_text:
                st.warning(f"Moderator prompt '{suffix}' not found.")
                break
            
            # Update prompt
            prompt_text = (
                prompt_text
                .replace("{chat_history}", chat_history)
                .replace("{time_spent}", str(time_spent))
                .replace("{speaking_time}", speaking_time_str)
                .replace("{transition_count}", str(transition_count))
            )

            # Debug: show visible moderator prompt
            if DEBUGGING == True:
                st.subheader("🔍 Visible Moderator Prompt")
                st.code(prompt_text)

            # Call AI model
            if api == "openai":
                response_vis = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "assistant", "content": prompt_text}],
                    temperature=TEMPERATURE_VIS_MOD,
                    **token_params
                )
                vis_msg = response_vis.choices[0].message.content.strip()

            elif api == "ollama":
                response_vis = requests.post(
                    url="http://localhost:11434/api/chat",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "assistant", "content": prompt_text}],
                        "stream": False, # Note that there is no real option to use streaming here. If streaming is set to "True" the response is returned in chunks which the following code is not able to return. If you want to use streaming you have to adjust the code.
                        "temperature": TEMPERATURE_VIS_MOD,
                        "max_tokens": MAX_OUTPUT_TOKENS,
                    }
                )
                data_vis = response_vis.json()
                vis_msg = data_vis["message"]["content"].strip()

            # Debug: show visible moderator output
            if DEBUGGING == True:
                st.subheader("📤 Visible Moderator Output")
                st.code(response_vis)
                st.code(vis_msg)
            
            ### Closing code check
            # Conversation is not closed
            if not any(
                code in vis_msg for code in CLOSING_MESSAGES.keys()
            ):
                    
                    # Display AI model output as chat message
                    with st.chat_message(ROLES["Moderator"]["display_role"], avatar=ROLES["Moderator"]["avatar"]):
                        st.markdown(f"**Moderator({NAME_MODERATOR}):** {vis_msg}")

                        # Append message to log files
                        st.session_state.messages.append({"role": "Moderator", "content": vis_msg})
                        st.session_state.messages_full.append({"role": "Moderator", "content": vis_msg})

                    # Save progress as backup, ignore errors.
                    try:
                        save_focusgroup_data(
                        st.session_state.backups_paths["transcript"],
                        st.session_state.backups_paths["transcript_full"],
                        st.session_state.backups_paths["time"],
                        )
                    except:
                        pass

            # Conversation is closed
            for code, closing_text in CLOSING_MESSAGES.items():
                if code in vis_msg:
                    st.session_state.interview_active = False

                    # Display chat message with closing text
                    with st.chat_message(ROLES["Moderator"]["display_role"], avatar=ROLES["Moderator"]["avatar"]):
                        st.markdown(f"**Moderator({NAME_MODERATOR}):** {closing_text}")
                    
                    # Append message to log files
                    st.session_state.messages.append({"role": "Moderator", "content": closing_text})
                    st.session_state.messages_full.append({"role": "Moderator", "content": closing_text})
                    
                    # Save final transcript and timing data, retry until successful.
                    if st.session_state.username.strip().lower() == "testaccount":
                        save_focusgroup_data(
                            st.session_state.transcript_paths["transcript"],
                            st.session_state.transcript_paths["transcript_full"],
                            st.session_state.transcript_paths["time"],
                        )
                    else: 
                        final_transcript_stored = False
                        while final_transcript_stored == False:
                            save_focusgroup_data(
                                st.session_state.transcript_paths["transcript"],
                                st.session_state.transcript_paths["transcript_full"],
                                st.session_state.transcript_paths["time"],
                            )
                            final_transcript_stored = check_if_focusgroup_completed(
                            TRANSCRIPTS_DIRECTORY, st.session_state.username
                            )
                    time.sleep(0.1)
                    break
                    


