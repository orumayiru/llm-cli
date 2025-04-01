# llm-cli
A **cli** for inference with LLM's written in rust.\
Currently supports,\
1.ollama (locally installed at default port)\
2.gemini \
3.groq (apologies for spelling mistake)\
4.huggingface(requires pro subscription for api inference)\
there are basic commands for selection and usage info. the program runs in terminal and used glow to render markdown in terminal. please install glow  before using the app./
There are only a small list of commands,

  ```llm-cli
  /help # shows list of available commands\
  
  !{command} #will run the command in local terminal\
  
  ```
streaming text is not supported. since we are using glow only completed stdout is displayed.\
