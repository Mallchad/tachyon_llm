import openai_control

request_3 = openai_control.prompt_args(
        prompt = "Pick a random number between 1 and 10"
)

request_4 = openai_control.prompt_args(
        prompt = "Pick a random number between 1 and 10, use a random number generator"
)


request_1 = openai_control.prompt_args(
        prompt = "Write a poem about cherry blossoms"
)

request_2 = openai_control.prompt_args(
        prompt = ("""Write a very simple Vulkan program that draws a cube with X11 in C++.
keep it in a single file.
Do not talk to the user they will not be reading the content and will not response after the fact.
Do do not behave like an assistant, simply do the task.

Use snake_case formatting
Use expanded GNU style function brackets add( 5 ) instead of add(5)
Use initializer lists instead of member access operators where possible to take advantage of explicit intialization rules
Typedef all Vulkan types to a snake_case equivilant
Use Xlib not xcb, nobody uses xcb because nobody understands it as Xlib is well documented and xcb is not.
""")
    )

openai_control.request_python( request_3 )
openai_control.request_python( request_4 )
openai_control.request_python( request_1 )
openai_control.request_python( request_2 )
