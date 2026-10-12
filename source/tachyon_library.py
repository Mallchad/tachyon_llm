# transient/python_container/bin/python3 -m mypy source/tachyon_library.py  --strict

import os
import subprocess
import sys
import time

dry_run: bool = False

def script_location():
    # Self-Script Information
    result = []
    result.script_name = os.path.basename(sys.argv[0])
    result.script_dir_relative = os.path.dirname(sys.argv[0])
    result.script_dir_absolute = os.path.realpath(script_dir_relative)


# Self-Script Information
script_name = os.path.basename(sys.argv[0])
script_dir_relative = os.path.dirname(sys.argv[0])
script_dir_absolute = os.path.realpath(script_dir_relative)

def tyon_log(*vargs: object) -> None:
    # Convert to nanoseconds
    current_time = int(time.time() * 1e9)
    display_time = str( current_time )
    print( f"[{display_time}][Tachyon Build Py]", *vargs)

def tyon_error(*vargs: object) -> None:
	print("[Tachyon Build Py][ERROR]", *vargs)

def tyon_debug(*vargs) -> None:
	print("[DEBUG]", *vargs)

# Always start from project root location
os.chdir(script_dir_absolute)
os.system("")  # enables ansi escape characters in terminal
tyon_log(f"Changing directory to '{script_dir_absolute}'")
tyon_log(f"Running script {script_name}")

def log_exception_generic( err, log_message ) -> None:
    tyon_log(f"""{log_message}"
            Type: {type(err).__name__}
            Message: {err} """)

def execute( command ):
    if dry_run:
        tyon_log( f"(dry run) executing command: \"{command}\"" )
        return 0
    else:
        tyon_log( f"executing command: \"{command}\"" )
        completed = None
        # check = false silences exception on command non-successful exit but not other exceptions
        try: completed = subprocess.run( command.split(), check = False )
        except Exception as err: log_exception_generic( err, "Unknown Exception" )

        if completed == None or completed.returncode != 0:
            tyon_error( "Failed to execute command" )
        return (completed and completed.returncode or 1)

def execute_list( command_list: list ) -> int:
    commands = None
    if type( command_list[0] ) == "list":
        commands = command_list[0]
    else:
        commands = [*command_list]
    if dry_run:
        tyon_log( f"(dry run) executing command: \"{commands}\"" )
        return 0
    else:
        tyon_log( f"executing command: '{commands}'" )
        completed = None
        # check = false silences exception on command non-successful exit but not other exceptions
        try: completed = subprocess.run( commands, check = False )
        except Exception as err: log_exception_generic( err, "Unknown Exception" )
        if completed == None or completed.returncode != 0:
            tyon_error( "Failed to execute command" )
        return (completed and completed.returncode or 1)

# Same as previous but consume input from pipe (only works once per script)
def execute_list_piped( *command_list ):
    commands = None
    if type( command_list[0] ) == "list":
        commands = command_list[0]
    else:
        commands = [*command_list]
    if dry_run:
        tyon_log( f"(dry run) executing command: \"{commands}\"" )
        return 0
    else:
        tyon_log( f"executing command: '{commands}'" )
        # check = false silences exception on command non-successful exit but not other exceptions
        try: completed = subprocess.run( commands, stdin=sys.stdin.buffer, check = False )
        except Exception as err: log_exception_generic( err, "Unknown Exception" )
        if completed == None or completed.returncode != 0:
            tyon_error( "Failed to execute command" )
        return completed.returncode
