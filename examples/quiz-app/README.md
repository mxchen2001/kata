# Quiz App — a school project built with Kata

A solar system quiz app generated entirely from `.kata` files.

## What it builds

A Python CLI that quizzes you on the solar system — 10 multiple choice questions with scoring and explanations.

## Files

| File | Generates |
|---|---|
| `1-questions.kata` | `questions.json` — 10 quiz questions with answers |
| `2-app.kata` | `quiz.py` — CLI app that runs the quiz |
| `3-tests.kata` | `test_quiz.py` — test suite |

## Run

```sh
kata exec examples/quiz-app
```

Then try it:

```sh
cd output/quiz-app
python quiz.py
```
