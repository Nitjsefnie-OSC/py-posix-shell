from __future__ import annotations

import base64
import datetime as dt
import io
import os
import sys

import pytest

import py_posix_shell.posix_utils as posix_utils
import py_posix_shell.shell as shell_module
from py_posix_shell import cli
from py_posix_shell.lexer import dump_tokens, lex
from py_posix_shell.errors import ShellExit
from py_posix_shell.parser import parse
from py_posix_shell.posix_utils import ProcessInfo, StorageVolume, WindowsViEditor
from py_posix_shell.shell import LineHistoryState, Shell, terminal_display_width


def run_shell(source: str, **kwargs):
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(stdout=stdout, stderr=stderr, **kwargs)
    status = shell.execute(source)
    return status, stdout.getvalue(), stderr.getvalue(), shell


class TtyStringIO(io.StringIO):
    def isatty(self):
        return True


def test_lexer_keeps_quotes_and_operators():
    assert dump_tokens(lex("echo 'a b' \"$HOME\" && false")) == [
        "word:echo",
        "word:a b",
        "word:$HOME",
        "op:&&",
        "word:false",
    ]


def test_lexer_preserves_windows_current_directory_backslash():
    if os.name != "nt":
        return

    assert dump_tokens(lex(r".\build_windows.bat")) == [r"word:.\build_windows.bat"]
    assert dump_tokens(lex(r"..\tools\build.cmd")) == [r"word:..\tools\build.cmd"]


@pytest.mark.parametrize(
    ("operator", "continuation"),
    [
        ("&&", "\n"),
        ("&&", "\n\n"),
        ("&&", " # comment\n"),
        ("||", "\n"),
        ("||", "\n\n"),
        ("||", " # comment\n"),
        ("|", "\n"),
        ("|", "\n\n"),
        ("|", " # comment\n"),
    ],
)
def test_linebreak_after_continuation_operator_matches_inline(operator, continuation):
    assert parse(f"a {operator} b") == parse(f"a {operator}{continuation}b")


def test_unadorned_newline_still_separates_commands():
    assert len(parse("a\nb").items) == 2


def test_assignment_and_parameter_expansion():
    status, stdout, stderr, _shell = run_shell("name=world; echo hello-$name")
    assert status == 0
    assert stdout == "hello-world\n"
    assert stderr == ""


def test_shell_environment_variable_is_set_and_exported(tmp_path):
    pysh_path = tmp_path / ("pysh.exe" if os.name == "nt" else "pysh")
    shell = Shell(env={"PATH": ""}, argv0="script.sh", shell_path=str(pysh_path))

    assert shell.get_parameter("0") == "script.sh"
    assert os.path.normcase(shell.get_parameter("SHELL")) == os.path.normcase(os.path.abspath(pysh_path))
    assert shell.env["SHELL"] == shell.get_parameter("SHELL")


def test_shell_environment_variable_preserves_existing_value():
    shell = Shell(env={"PATH": "", "SHELL": "/bin/custom"}, shell_path="/tmp/pysh")

    assert shell.get_parameter("SHELL") == "/bin/custom"


def test_cli_sets_shell_to_pysh_entry_when_running_script(monkeypatch, tmp_path, capsys):
    script = tmp_path / "show-shell.pysh"
    script.write_text('printf "%s\\n%s\\n" "$0" "$SHELL"\n', encoding="utf-8")
    pysh_path = tmp_path / ("pysh.exe" if os.name == "nt" else "pysh")
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setattr(sys, "argv", [str(pysh_path), str(script)])

    status = cli.main([str(script)])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert lines[0] == str(script)
    assert os.path.normcase(lines[1]) == os.path.normcase(os.path.abspath(pysh_path))


def test_startup_file_sources_pyshrc_into_current_shell(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".pyshrc").write_text("rc_value=loaded\nalias showrc='echo rc:$rc_value'\n", encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(env={"HOME": str(home), "PATH": ""}, stdout=stdout, stderr=stderr)

    assert shell.source_startup_file() == 0
    status = shell.execute("showrc")
    assert status == 0
    assert stdout.getvalue() == "rc:loaded\n"
    assert stderr.getvalue() == ""


def test_startup_file_missing_is_noop(tmp_path):
    shell = Shell(env={"HOME": str(tmp_path), "PATH": ""}, stdout=io.StringIO(), stderr=io.StringIO())

    assert shell.source_startup_file() == 0
    assert shell.stdout.getvalue() == ""
    assert shell.stderr.getvalue() == ""


def test_cli_sources_pyshrc_before_interactive_repl(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".pyshrc").write_text("rc_value=from-rc\n", encoding="utf-8")
    pysh_path = tmp_path / ("pysh.exe" if os.name == "nt" else "pysh")

    def fake_repl(self):
        self.stdout.write(self.get_parameter("rc_value") + "\n")
        return 0

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(sys, "argv", [str(pysh_path)])
    monkeypatch.setattr(shell_module.Shell, "repl", fake_repl)

    status = cli.main(["-i"])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.out == "from-rc\n"
    assert captured.err == ""


def test_and_or_status():
    status, stdout, _stderr, _shell = run_shell("false && echo no; false || echo yes")
    assert status == 0
    assert stdout == "yes\n"


def test_and_or_updates_status_before_expanding_next_command():
    status, stdout, stderr, _shell = run_shell('false || echo "or:$?"; true && echo "and:$?"')

    assert status == 0
    assert stdout == "or:1\nand:0\n"
    assert stderr == ""


def test_redirection(tmp_path):
    path = tmp_path / "out.txt"
    status, stdout, stderr, _shell = run_shell(f'echo saved > "{path}"')
    assert status == 0
    assert stdout == ""
    assert stderr == ""
    assert path.read_text(encoding="utf-8") == "saved\n"


def test_pipeline_with_builtin_and_external_python():
    code = "import sys; print(sys.stdin.read().upper(), end='')"
    status, stdout, stderr, _shell = run_shell(f'echo hello | "{sys.executable}" -c "{code}"')
    assert status == 0
    assert stdout == "HELLO\n"
    assert stderr == ""


def test_command_substitution():
    status, stdout, stderr, _shell = run_shell("echo $(printf hi)")
    assert status == 0
    assert stdout == "hi\n"
    assert stderr == ""


def test_parameter_defaults_and_escaped_dollar():
    status, stdout, stderr, _shell = run_shell(r'echo ${missing:-fallback}; name=${name:=set}; echo "\$name" "$name"')
    assert status == 0
    assert stdout == "fallback\n$name set\n"
    assert stderr == ""


def test_cd_updates_pwd(tmp_path):
    old = os.getcwd()
    try:
        status, stdout, stderr, shell = run_shell(f'cd "{tmp_path}"; pwd')
        assert status == 0
        assert stdout == str(tmp_path) + "\n"
        assert stderr == ""
        assert shell.get_parameter("PWD") == str(tmp_path)
    finally:
        os.chdir(old)


def test_if_for_while_and_arithmetic():
    script = """
if false; then
  echo bad
elif true; then
  echo good
else
  echo bad
fi

for item in a b; do
  echo "for:$item"
done

i=0
while [ "$i" -lt 3 ]; do
  i=$((i + 1))
  [ "$i" -eq 2 ] && continue
  echo "while:$i"
done
"""
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 0
    assert stdout == "good\nfor:a\nfor:b\nwhile:1\nwhile:3\n"
    assert stderr == ""


def test_until_case_function_and_return():
    script = """
say() {
  case "$1" in
    a|b) echo "letter:$1" ;;
    *) echo other ;;
  esac
  return 7
}
say b
echo "status:$?"

i=0
until [ "$i" -eq 2 ]; do
  i=$((i + 1))
done
echo "until:$i"
"""
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 0
    assert stdout == "letter:b\nstatus:7\nuntil:2\n"
    assert stderr == ""


def test_local_builtin_scopes_function_variables():
    script = r"""
value=global
scratch=global-scratch
f() {
  local value=inner scratch=temp
  echo "f:$value:$scratch"
}
f
echo "after:$value:$scratch"

blank=outer
g() {
  local blank
  echo "blank:${blank-unset}:${blank:-fallback}"
  blank=changed
}
g
echo "blank-after:$blank"

nested_outer() {
  local value=outer
  nested_inner() {
    local value=inner
    echo "nested:$value"
  }
  nested_inner
  echo "outer:$value"
}
nested_outer
echo "final:$value"
"""
    status, stdout, stderr, _shell = run_shell(script)

    assert status == 0
    assert stdout == (
        "f:inner:temp\n"
        "after:global:global-scratch\n"
        "blank::fallback\n"
        "blank-after:outer\n"
        "nested:inner\n"
        "outer:outer\n"
        "final:global\n"
    )
    assert stderr == ""


def test_local_builtin_requires_function_scope():
    status, stdout, stderr, _shell = run_shell("local value=bad")

    assert status == 1
    assert stdout == ""
    assert stderr == "local: can only be used in a function\n"


def test_backslash_local_supports_conda_hook_style_functions():
    script = r"""
__conda_activate() {
  \local ask_conda
  ask_conda=hooked
  echo "$ask_conda"
}
conda() {
  \local cmd="${1-__missing__}"
  case "$cmd" in
    activate) __conda_activate "$@" ;;
    *) echo "cmd:$cmd" ;;
  esac
}
cmd=global
conda activate
echo "after:$cmd:${ask_conda-unset}"
"""
    status, stdout, stderr, _shell = run_shell(script)

    assert status == 0
    assert stdout == "hooked\nafter:global:unset\n"
    assert stderr == ""


def test_double_bracket_conditionals_support_shell_semantics(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("data\n", encoding="utf-8")
    script = f'''
name=hello.txt
empty=
if [[ $name == *.txt && $name =~ ^hello[.] ]]; then echo pattern-regex; fi
if [[ $name == "*.txt" ]]; then echo bad-literal; else echo literal-ok; fi
if [[ ! ( 3 -lt 2 || -z "$name" ) ]]; then echo logic-ok; fi
if [[ -n "$name" && "$name" != *.py && apple < banana ]]; then echo strings-ok; fi
if [[ -f "{sample}" && -s "{sample}" ]]; then echo file-ok; fi
if [[ $empty ]]; then echo bad-empty; else echo empty-ok; fi
'''
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})

    assert status == 0
    assert stdout == "pattern-regex\nliteral-ok\nlogic-ok\nstrings-ok\nfile-ok\nempty-ok\n"
    assert stderr == ""


def test_double_bracket_reports_syntax_errors():
    status, stdout, stderr, _shell = run_shell("[[ a && ]]")

    assert status == 2
    assert stdout == ""
    assert "[[:" in stderr


def test_here_documents_and_parameter_operators():
    script = r"""
name=world
while read line; do echo "$line"; done <<EOF
hello $name
EOF
while read line; do echo "$line"; done <<'EOF'
literal $name
EOF
path=/usr/local/bin
echo "${#name}:${path##*/}:${path%/*}"
"""
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 0
    assert stdout == "hello world\nliteral $name\n5:bin:/usr/local\n"
    assert stderr == ""


def test_parameter_replacement_word_removes_syntax_quotes():
    status, stdout, stderr, _shell = run_shell(r'PATH=old; echo "${PATH:+":${PATH}"}"', env={"PATH": ""})

    assert status == 0
    assert stdout == ":old\n"
    assert stderr == ""


def test_double_quoted_command_substitution_allows_nested_quotes():
    status, stdout, stderr, _shell = run_shell(r'value="$(printf "%s" "$(printf inner)")"; echo "$value"', env={"PATH": ""})

    assert status == 0
    assert stdout == "inner\n"
    assert stderr == ""


def test_conda_style_path_expression_with_nested_command_substitution():
    script = r'''
CONDA_EXE=/c/ProgramData/miniforge3/Scripts/conda.exe
PATH=base
PATH="$(dirname "$(dirname "$CONDA_EXE")")/condabin${PATH:+":${PATH}"}"
echo "$PATH"
'''
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})

    assert status == 0
    assert stdout == "/c/ProgramData/miniforge3/condabin:base\n"
    assert stderr == ""


def test_dot_eval_and_quoted_at(tmp_path):
    script_file = tmp_path / "lib.sh"
    script_file.write_text("from_dot=ok\nreturn 4\n", encoding="utf-8")
    script = f'''
. "{script_file}"
echo "$from_dot:$?"
set -- "a b" c
for arg in "$@"; do echo "arg:$arg"; done
eval 'echo eval:$from_dot'
'''
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 0
    assert stdout == "ok:4\narg:a b\narg:c\neval:ok\n"
    assert stderr == ""


def test_source_accepts_windows_msys_and_cygdrive_paths(tmp_path):
    if os.name != "nt":
        return

    script_file = tmp_path / "lib.sh"
    script_file.write_text("from_msys=ok\n", encoding="utf-8")
    drive, tail = os.path.splitdrive(str(script_file))
    msys_path = "/" + drive[:1].lower() + tail.replace("\\", "/")
    cygdrive_path = "/cygdrive/" + drive[:1].lower() + tail.replace("\\", "/")

    status, stdout, stderr, _shell = run_shell(
        f'''
. "{msys_path}"
echo "$from_msys"
unset from_msys
source "{cygdrive_path}"
echo "$from_msys"
''',
        env={"PATH": ""},
    )

    assert status == 0
    assert stdout == "ok\nok\n"
    assert stderr == ""


def test_unset_f_removes_shell_function_and_command_queries():
    script = """
__pysh_activate_fn() { :; }
__pysh_keep_fn() { :; }
__pysh_var=value
command -v __pysh_activate_fn
command -V __pysh_activate_fn
type __pysh_activate_fn
unset -f __pysh_activate_fn
command -v __pysh_activate_fn || echo gone
unset -f __pysh_keep_fn -v __pysh_var
command -v __pysh_keep_fn || echo keep-gone
echo "${__pysh_var:-var-gone}"
"""
    status, stdout, stderr, _shell = run_shell(script)

    assert status == 0
    assert stdout == (
        "__pysh_activate_fn\n"
        "__pysh_activate_fn is a shell function\n"
        "__pysh_activate_fn is a shell function\n"
        "gone\n"
        "keep-gone\n"
        "var-gone\n"
    )
    assert stderr == ""


def test_source_windows_venv_activate_script_updates_path_and_deactivate(tmp_path):
    if os.name != "nt":
        return

    venv = tmp_path / "venv"
    scripts_dir = venv / "Scripts"
    scripts_dir.mkdir(parents=True)
    fake_python = scripts_dir / "python.exe"
    fake_python.write_text("", encoding="utf-8")
    venv_text = str(venv)
    quoted_venv = venv_text.replace("'", "'\"'\"'")
    activate = tmp_path / "activate"
    activate.write_text(
        f"""
deactivate () {{
    if [ -n "${{_OLD_VIRTUAL_PATH:-}}" ] ; then
        PATH="${{_OLD_VIRTUAL_PATH:-}}"
        export PATH
        unset _OLD_VIRTUAL_PATH
    fi
    if [ -n "${{_OLD_VIRTUAL_PYTHONHOME:-}}" ] ; then
        PYTHONHOME="${{_OLD_VIRTUAL_PYTHONHOME:-}}"
        export PYTHONHOME
        unset _OLD_VIRTUAL_PYTHONHOME
    fi
    hash -r 2> /dev/null
    if [ -n "${{_OLD_VIRTUAL_PS1:-}}" ] ; then
        PS1="${{_OLD_VIRTUAL_PS1:-}}"
        export PS1
        unset _OLD_VIRTUAL_PS1
    fi
    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT
    if [ ! "${{1:-}}" = "nondestructive" ] ; then
        unset -f deactivate
    fi
}}

deactivate nondestructive

case "$(uname)" in
    CYGWIN*|MSYS*|MINGW*)
        VIRTUAL_ENV=$(cygpath '{quoted_venv}')
        export VIRTUAL_ENV
        ;;
    *)
        export VIRTUAL_ENV='{quoted_venv}'
        ;;
esac

_OLD_VIRTUAL_PATH="$PATH"
PATH="$VIRTUAL_ENV/"Scripts":$PATH"
export PATH
VIRTUAL_ENV_PROMPT=venv
export VIRTUAL_ENV_PROMPT
if [ -n "${{PYTHONHOME:-}}" ] ; then
    _OLD_VIRTUAL_PYTHONHOME="${{PYTHONHOME:-}}"
    unset PYTHONHOME
fi
if [ -z "${{VIRTUAL_ENV_DISABLE_PROMPT:-}}" ] ; then
    _OLD_VIRTUAL_PS1="${{PS1:-}}"
    PS1="("venv") ${{PS1:-}}"
    export PS1
fi
hash -r 2> /dev/null
""",
        encoding="utf-8",
    )

    command = f'''
source "{activate}"
command -v python
command -V deactivate
deactivate
printf 'after:%s:%s:%s:%s\\n' "$?" "${{VIRTUAL_ENV:-gone}}" "${{VIRTUAL_ENV_PROMPT:-gone}}" "${{PYTHONHOME:-gone}}"
printf 'ps1:%s\\n' "$PS1"
command -v deactivate || echo deactivate-gone
'''
    status, stdout, stderr, _shell = run_shell(
        command,
        env={"PATH": "", "PATHEXT": ".EXE;.BAT;.CMD", "PS1": "$ ", "PYTHONHOME": "oldhome"},
    )
    lines = stdout.splitlines()

    assert status == 0
    assert os.path.normcase(lines[0]) == os.path.normcase(str(fake_python))
    assert lines[1] == "deactivate is a shell function"
    assert lines[2] == "after:0:gone:gone:oldhome"
    assert lines[3] == "ps1:$ "
    assert lines[4] == "deactivate-gone"
    assert stderr == ""


def test_windows_cygpath_fallback_converts_basic_paths():
    if os.name != "nt":
        return

    status, stdout, stderr, _shell = run_shell(
        r"cygpath 'C:\Users\neko\venv'; cygpath -w /c/Users/neko/venv; cygpath -m 'C:\Users\neko\venv'",
        env={"PATH": ""},
    )
    lines = stdout.splitlines()

    assert status == 0
    assert lines[0] == "/c/Users/neko/venv"
    assert os.path.normcase(lines[1]) == os.path.normcase(r"C:\Users\neko\venv")
    assert lines[2] == "C:/Users/neko/venv"
    assert stderr == ""


def test_windows_cygpath_fallback_converts_path_lists():
    if os.name != "nt":
        return

    status, stdout, stderr, _shell = run_shell(
        r"cygpath --unix --path 'C:\Users\neko\venv;D:\tools'; cygpath --windows --path /c/Users/neko/venv:/d/tools",
        env={"PATH": ""},
    )
    lines = stdout.splitlines()

    assert status == 0
    assert lines[0] == "/c/Users/neko/venv:/d/tools"
    assert os.path.normcase(lines[1]) == os.path.normcase(r"C:\Users\neko\venv;D:\tools")
    assert stderr == ""


def test_windows_shell_adds_packaged_cygpath_to_child_path(tmp_path):
    if os.name != "nt":
        return

    bindir = tmp_path / "Scripts"
    bindir.mkdir()
    pysh = bindir / "pysh.exe"
    cygpath = bindir / "cygpath.exe"
    pysh.write_text("", encoding="utf-8")
    cygpath.write_text("", encoding="utf-8")

    shell = Shell(env={"PATH": r"C:\base"}, shell_path=str(pysh))

    path_entries = {os.path.normcase(entry) for entry in shell.env["PATH"].split(os.pathsep)}
    assert os.path.normcase(str(bindir)) in path_entries


def test_windows_conda_posix_environment_ignores_bash_without_cygpath(tmp_path):
    if os.name != "nt":
        return

    bindir = tmp_path / "Scripts"
    broken_bash_dir = tmp_path / "WindowsApps"
    bindir.mkdir()
    broken_bash_dir.mkdir()
    pysh = bindir / "pysh.exe"
    cygpath = bindir / "cygpath.exe"
    conda = tmp_path / "conda.exe"
    bash = broken_bash_dir / "bash.exe"
    for path in (pysh, cygpath, conda, bash):
        path.write_text("", encoding="utf-8")
    shell = Shell(env={"PATH": str(broken_bash_dir)}, shell_path=str(pysh))

    prepared = shell_module.prepare_external_environment(str(conda), [str(conda), "shell.posix", "activate", "base"], shell.env)
    path_entries = {os.path.normcase(entry) for entry in prepared["PATH"].split(os.pathsep)}

    assert os.path.normcase(str(bindir)) in path_entries
    assert os.path.normcase(str(broken_bash_dir)) not in path_entries


def test_group_subshell_negation_and_set_options(tmp_path):
    (tmp_path / "match.txt").write_text("", encoding="utf-8")
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        script = """
{ echo grouped; } > grouped.out
( x=changed; echo "sub:$x" )
echo "parent:${x:-empty}"
! false
echo "not:$?"
set -f
echo *.txt
set +f
echo *.txt
set -u
echo "$missing"
"""
        status, stdout, stderr, _shell = run_shell(script)
        assert status == 2
        assert stdout == "sub:changed\nparent:empty\nnot:0\n*.txt\nmatch.txt\n"
        assert "missing: parameter not set" in stderr
        assert (tmp_path / "grouped.out").read_text(encoding="utf-8") == "grouped\n"
    finally:
        os.chdir(old)


def test_set_e_exits_on_uncontrolled_failure_but_not_and_or():
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(stdout=stdout, stderr=stderr)
    try:
        shell.execute("set -e; false && echo skipped; echo alive; false; echo dead")
    except ShellExit as exc:
        assert exc.status == 1
    else:
        raise AssertionError("set -e did not exit")
    assert stdout.getvalue() == "alive\n"
    assert stderr.getvalue() == ""


def test_alias_unalias_and_arithmetic_side_effects():
    script = """
alias hi='echo hello'
hi world
unalias hi
i=1
echo "$((i++)):$i"
echo "$((++i)):$i"
echo "$((i += 5)):$i"
"""
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 0
    assert stdout == "hello world\n1:2\n3:3\n8:8\n"
    assert stderr == ""


def test_trap_exit_runs_with_original_status():
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(stdout=stdout, stderr=stderr)
    try:
        shell.execute("trap 'echo exit:$?' EXIT; echo body; exit 7")
    except ShellExit as exc:
        assert exc.status == 7
    else:
        raise AssertionError("exit did not raise")
    assert stdout.getvalue() == "body\nexit:7\n"
    assert stderr.getvalue() == ""


def test_getopts_hash_umask_times_and_more_test_ops(tmp_path):
    data = tmp_path / "data.txt"
    link = tmp_path / "data-link.txt"
    devnull = os.devnull
    data.write_text("x", encoding="utf-8")
    try:
        link.symlink_to(data)
    except OSError:
        link = data
    script = f'''
set -- -a -b value rest
while getopts ab: opt; do
  echo "opt:$opt:$OPTARG:$OPTIND"
done
[ -s "{data}" ] && echo sized
[ "{data}" -ef "{link}" ] && echo same
[ "{data}" -nt "{tmp_path / 'missing'}" ] || echo nt-missing
hash "{sys.executable}" || true
umask > "{devnull}"
times > "{devnull}"
'''
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 0
    assert "opt:a::2\n" in stdout
    assert "opt:b:value:4\n" in stdout
    assert "sized\n" in stdout
    assert "same\n" in stdout
    assert "nt-missing\n" in stdout
    assert stderr == ""


def test_read_write_redirection_and_readonly(tmp_path):
    path = tmp_path / "rw.txt"
    path.write_text("abc", encoding="utf-8")
    script = f'''
exec 3<> "{path}"
readonly locked=one
locked=two
'''
    status, stdout, stderr, _shell = run_shell(script)
    assert status == 2
    assert stdout == ""
    assert "locked: is read only" in stderr


def test_internal_posix_utilities_work_without_path(tmp_path):
    env = {"PATH": ""}
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    moved = tmp_path / "moved.txt"
    nested = tmp_path / "a" / "b"
    src.write_text("alpha\nbeta\nalpha-two\n", encoding="utf-8")
    script = f'''
mkdir -p "{nested}"
cp "{src}" "{dst}"
mv "{dst}" "{moved}"
printf "extra\\n" >> "{moved}"
ls -1 "{tmp_path}"
cat "{moved}" | grep alpha | wc -l
head -n 1 "{moved}"
tail -n 1 "{moved}"
touch "{tmp_path / 'new.txt'}"
rm "{tmp_path / 'new.txt'}"
rmdir "{nested}"
rmdir "{nested.parent}"
basename "{moved}"
dirname "{moved}"
'''
    status, stdout, stderr, _shell = run_shell(script, env=env)
    assert status == 0
    assert "moved.txt" in stdout
    assert "\n2\n" in stdout
    assert "alpha\n" in stdout
    assert "extra\n" in stdout
    assert "moved.txt\n" in stdout
    assert str(tmp_path) in stdout
    assert not (tmp_path / "new.txt").exists()
    assert not nested.parent.exists()
    assert stderr == ""


def test_cat_decodes_gbk_file_when_stdout_is_tty(tmp_path):
    text = chr(0x4F60) + chr(0x597D) + "\n"
    path = tmp_path / "gbk.txt"
    path.write_bytes(text.encode("gbk"))
    stdout = TtyStringIO()
    stderr = io.StringIO()
    shell = Shell(stdout=stdout, stderr=stderr, env={"PATH": ""})

    status = shell.execute(f'cat "{path}"')

    assert status == 0
    assert stdout.getvalue() == text
    assert stderr.getvalue() == ""


def test_cat_preserves_bytes_when_piped_or_redirected(tmp_path):
    text = chr(0x4F60) + chr(0x597D) + "\n"
    data = text.encode("gbk")
    source = tmp_path / "gbk.txt"
    copy = tmp_path / "copy.txt"
    source.write_bytes(data)

    status, stdout, stderr, _shell = run_shell(
        f'cat "{source}" | wc -c\ncat "{source}" > "{copy}"',
        env={"PATH": ""},
    )

    assert status == 0
    assert stdout == f"{len(data)}\n"
    assert copy.read_bytes() == data
    assert stderr == ""


def test_text_pipeline_still_uses_utf8_bytes_for_binary_readers():
    text = chr(0x4E2D) + chr(0x6587)

    status, stdout, stderr, _shell = run_shell(f'echo "{text}" | wc -c', env={"PATH": ""})

    assert status == 0
    assert stdout == f"{len((text + chr(10)).encode('utf-8'))}\n"
    assert stderr == ""


def test_internal_process_and_identity_utilities_without_path():
    command = "ps; uname -s; whoami; id; date +%Y" if os.name == "nt" else "uname -s; whoami; id; date +%Y"
    status, stdout, stderr, _shell = run_shell(
        command,
        env={"PATH": ""},
    )
    assert status == 0
    if os.name == "nt":
        assert stdout.splitlines()[0].split() == ["PID", "TTY", "TIME", "CMD"]
        assert stdout.count("\n") >= 5
    else:
        assert stdout.count("\n") >= 4
    assert stderr == ""


def test_windows_uname_fallback_matches_msys_style(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(posix_utils.os, "name", "nt")
    monkeypatch.setattr(posix_utils.platform, "node", lambda: "host")
    monkeypatch.setattr(posix_utils.platform, "release", lambda: "11")
    monkeypatch.setattr(posix_utils.platform, "version", lambda: "10.0.26100")
    monkeypatch.setattr(posix_utils.platform, "machine", lambda: "AMD64")

    status = posix_utils.utility_uname(None, ["uname", "-s"], io.StringIO(), stdout, stderr)

    assert status == 0
    assert stdout.getvalue() == "MINGW64_NT-10.0.26100\n"
    assert stderr.getvalue() == ""


def test_windows_ps_formats_linux_style_process_views(monkeypatch):
    now = dt.datetime.now()
    current_pid = os.getpid()
    fake_processes = [
        ProcessInfo(
            pid=current_pid,
            ppid=10,
            user="DESKTOP\\neko",
            tty="?",
            cpu_seconds=65,
            start_time=now - dt.timedelta(minutes=10),
            cmd="pysh.exe",
            name="pysh.exe",
            rss=50 * 1024 * 1024,
            virtual_size=100 * 1024 * 1024,
            state="R",
        ),
        ProcessInfo(
            pid=current_pid + 1,
            ppid=current_pid,
            user="DESKTOP\\neko",
            tty="?",
            cpu_seconds=3,
            start_time=now - dt.timedelta(minutes=2),
            cmd='python.exe child.py',
            name="python.exe",
            rss=20 * 1024 * 1024,
            virtual_size=80 * 1024 * 1024,
        ),
        ProcessInfo(
            pid=99999,
            ppid=4,
            user="NT AUTHORITY\\SYSTEM",
            tty="?",
            cpu_seconds=9,
            start_time=now - dt.timedelta(days=1),
            cmd="services.exe",
            name="services.exe",
            rss=10 * 1024 * 1024,
            virtual_size=60 * 1024 * 1024,
        ),
    ]

    monkeypatch.setattr(posix_utils.os, "name", "nt")
    monkeypatch.setattr(posix_utils, "collect_windows_processes", lambda *, detailed: fake_processes)
    monkeypatch.setattr(posix_utils, "windows_total_physical_memory", lambda: 200 * 1024 * 1024)

    status, stdout, stderr, _shell = run_shell("ps; ps -ef; ps aux; ps --help", env={"PATH": ""})

    assert status == 0
    assert stdout.splitlines()[0].split() == ["PID", "TTY", "TIME", "CMD"]
    default_section = stdout.split("UID", 1)[0]
    assert "pysh.exe" in default_section
    assert "python.exe" in default_section
    assert "services.exe" not in default_section
    assert "UID" in stdout
    assert "PPID" in stdout
    assert "STIME" in stdout
    assert "DESKTOP\\neko" in stdout
    assert "python.exe child.py" in stdout
    assert "USER" in stdout
    assert "%CPU" in stdout
    assert "%MEM" in stdout
    assert "COMMAND" in stdout
    assert "services.exe" in stdout
    assert "Usage: ps [aux|-ef]\n" in stdout
    assert stderr == ""


def test_top_fallback_batch_snapshot(monkeypatch):
    now = dt.datetime.now()
    fake_processes = [
        ProcessInfo(
            pid=42,
            ppid=1,
            user="neko",
            tty="?",
            cpu_seconds=12,
            start_time=now - dt.timedelta(minutes=5),
            cmd="python app.py",
            name="python",
            rss=64 * 1024 * 1024,
            virtual_size=128 * 1024 * 1024,
            state="R",
        ),
        ProcessInfo(
            pid=7,
            ppid=1,
            user="root",
            tty="?",
            cpu_seconds=1,
            start_time=now - dt.timedelta(minutes=10),
            cmd="service",
            name="service",
            rss=8 * 1024 * 1024,
            virtual_size=32 * 1024 * 1024,
            state="S",
        ),
    ]
    monkeypatch.setattr(posix_utils, "collect_top_processes", lambda *, detailed=True: fake_processes)
    monkeypatch.setattr(
        posix_utils,
        "memory_snapshot",
        lambda: {"total": 256 * 1024 * 1024, "free": 128 * 1024 * 1024, "used": 96 * 1024 * 1024, "cached": 32 * 1024 * 1024, "swap_total": 0, "swap_free": 0},
    )

    status, stdout, stderr, _shell = run_shell("top -b -n 1 -o pid", env={"PATH": ""})

    assert status == 0
    assert "top - " in stdout
    assert "Tasks:" in stdout
    assert "MiB Mem" in stdout
    assert "PID USER" in stdout
    assert "python app.py" in stdout
    assert "service" in stdout
    assert stderr == ""


def test_top_interactive_shows_loading_before_snapshot(monkeypatch):
    stdin = TtyStringIO()
    stdout = TtyStringIO()
    stderr = io.StringIO()

    def fake_render(options, *, interactive):
        del options
        assert interactive is True
        assert "top - collecting process snapshot..." in stdout.getvalue()
        return "top - ready\nPID USER COMMAND\n"

    monkeypatch.setattr(posix_utils, "render_top_snapshot", fake_render)
    monkeypatch.setattr(posix_utils, "wait_for_top_quit", lambda _stdin, _delay: False)

    status = posix_utils.utility_top(None, ["top", "-n", "1"], stdin, stdout, stderr)

    output = stdout.getvalue()
    assert status == 0
    assert "top - collecting process snapshot..." in output
    assert "top - ready" in output
    assert "\033[?1049h" not in output
    assert "\033[?1049l" not in output
    assert stderr.getvalue() == ""


def test_top_windows_uses_fast_process_snapshot(monkeypatch):
    now = dt.datetime.now()
    fake_processes = [
        ProcessInfo(
            pid=42,
            ppid=1,
            user="neko",
            tty="?",
            cpu_seconds=12,
            start_time=now,
            cmd="python app.py",
            name="python",
            rss=64 * 1024 * 1024,
            virtual_size=128 * 1024 * 1024,
            state="R",
        )
    ]
    detailed_flags = []

    monkeypatch.setattr(posix_utils.os, "name", "nt")
    monkeypatch.setattr(posix_utils, "collect_top_processes", lambda *, detailed=True: detailed_flags.append(detailed) or fake_processes)
    monkeypatch.setattr(
        posix_utils,
        "memory_snapshot",
        lambda: {"total": 256 * 1024 * 1024, "free": 128 * 1024 * 1024, "used": 96 * 1024 * 1024, "cached": 32 * 1024 * 1024, "swap_total": 0, "swap_free": 0},
    )

    interactive_snapshot = posix_utils.render_top_snapshot(posix_utils.TopOptions(), interactive=True)
    batch_snapshot = posix_utils.render_top_snapshot(posix_utils.TopOptions(), interactive=False)

    assert detailed_flags == [False, False]
    assert "python app.py" in interactive_snapshot
    assert "python app.py" in batch_snapshot
    assert "Press q to quit." in interactive_snapshot
    assert "Press q to quit." not in batch_snapshot


def test_ps_internal_fallback_is_windows_only(monkeypatch):
    monkeypatch.setattr(shell_module.os, "name", "posix")
    shell = Shell(env={"PATH": ""})

    assert shell.should_run_internal_utility("ps", shell.env) is False


def test_command_and_type_report_internal_utility_when_path_is_empty():
    status, stdout, stderr, _shell = run_shell("command -v ls; type ls", env={"PATH": ""})
    assert status == 0
    assert stdout.splitlines()[0] == "ls"
    assert "ls is a shell utility" in stdout
    assert stderr == ""


def test_windows_which_reports_executables_builtins_and_internal_utilities(tmp_path):
    if os.name != "nt":
        return
    first_bin = tmp_path / "first"
    second_bin = tmp_path / "second"
    first_bin.mkdir()
    second_bin.mkdir()
    first_tool = first_bin / "tool.exe"
    second_tool = second_bin / "tool.exe"
    first_tool.write_text("", encoding="utf-8")
    second_tool.write_text("", encoding="utf-8")

    status, stdout, stderr, _shell = run_shell(
        "which tool; which cd; which ls; which missing; echo status:$?; which -a tool",
        env={"PATH": f"{first_bin};{second_bin}", "PATHEXT": ".EXE;.CMD"},
    )

    lines = stdout.splitlines()
    assert status == 0
    assert os.path.normcase(lines[0]) == os.path.normcase(str(first_tool))
    assert lines[1] == "cd: shell built-in command"
    assert lines[2] == "ls: shell utility"
    assert lines[3] == "status:1"
    assert [os.path.normcase(line) for line in lines[4:]] == [
        os.path.normcase(str(first_tool)),
        os.path.normcase(str(second_tool)),
    ]
    assert stderr == ""


def test_windows_batch_files_run_from_current_directory(tmp_path):
    if os.name != "nt":
        return

    script = tmp_path / "runme.bat"
    script.write_bytes(b"@echo off\r\necho bat:%1\r\nexit /b 7\r\n")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        status, stdout, stderr, _shell = run_shell(
            r"""
.\runme.bat backslash
echo status:$?
./runme.bat slash
echo status:$?
runme.bat bare
echo status:$?
""",
            env={"PATH": "", "PATHEXT": ".EXE;.BAT;.CMD"},
            stdin=io.StringIO(""),
        )
    finally:
        os.chdir(old_cwd)

    assert status == 0
    assert stdout == "bat:backslash\nstatus:7\nbat:slash\nstatus:7\nstatus:127\n"
    assert stderr == "runme.bat: command not found\n"


def test_windows_bare_executable_searches_path_not_current_directory(tmp_path):
    if os.name != "nt":
        return

    local_tool = tmp_path / "local.exe"
    local_tool.write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    path_tool = bin_dir / "path-tool.exe"
    path_tool.write_text("", encoding="utf-8")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        local_shell = Shell(env={"PATH": "", "PATHEXT": ".EXE;.BAT;.CMD"})
        path_shell = Shell(env={"PATH": str(bin_dir), "PATHEXT": ".EXE;.BAT;.CMD"})

        assert local_shell.resolve_command("local.exe", local_shell.env) is None
        assert os.path.normcase(local_shell.resolve_command("./local.exe", local_shell.env) or "") == os.path.normcase(str(local_tool))
        assert os.path.normcase(local_shell.resolve_command(r".\local.exe", local_shell.env) or "") == os.path.normcase(str(local_tool))
        assert os.path.normcase(path_shell.resolve_command("path-tool.exe", path_shell.env) or "") == os.path.normcase(str(path_tool))
        assert os.path.normcase(path_shell.resolve_command("path-tool", path_shell.env) or "") == os.path.normcase(str(path_tool))
    finally:
        os.chdir(old_cwd)


def test_external_keyboard_interrupt_returns_130(monkeypatch):
    shell = Shell(stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO())
    shell.resolve_command = lambda _name, _env: "fake-more"  # type: ignore[method-assign]
    events: list[str] = []

    class FakeProcess:
        returncode = None

        def communicate(self, input=None):
            events.append(f"communicate:{input!r}")
            raise KeyboardInterrupt

        def wait(self, timeout=None):
            events.append(f"wait:{timeout}")
            if timeout == 0.2:
                raise shell_module.subprocess.TimeoutExpired("fake-more", timeout)
            self.returncode = -2
            return self.returncode

        def terminate(self):
            events.append("terminate")

        def send_signal(self, sig):
            events.append(f"signal:{sig}")

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(shell_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    assert shell.run_external(["more"]) == 130
    assert "communicate:''" in events
    assert "terminate" in events or any(event.startswith("signal:") for event in events)
    assert shell.stderr.getvalue() == ""


def test_repl_keyboard_interrupt_returns_to_prompt(monkeypatch):
    stdout = io.StringIO()
    shell = Shell(stdout=stdout, stderr=io.StringIO(), interactive=True)
    events = iter([KeyboardInterrupt, "echo after:$?", EOFError])

    def fake_input(_prompt):
        event = next(events)
        if isinstance(event, type) and issubclass(event, BaseException):
            raise event
        return event

    monkeypatch.setattr("builtins.input", fake_input)

    assert shell.repl() == 0
    assert shell.last_status == 0
    assert stdout.getvalue() == "\nafter:130\n\n"


def test_repl_syntax_error_reports_and_keeps_session_alive(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(stdout=stdout, stderr=stderr, interactive=True)
    events = iter([">", "echo |", "echo after:$?", EOFError])

    def fake_input(_prompt):
        event = next(events)
        if isinstance(event, type) and issubclass(event, BaseException):
            raise event
        return event

    monkeypatch.setattr("builtins.input", fake_input)

    assert shell.repl() == 0
    assert shell.last_status == 0
    assert stdout.getvalue() == "after:2\n\n"
    errors = stderr.getvalue()
    assert "pysh: syntax error:" in errors
    assert "expected word" in errors
    assert "expected command after pipe" in errors


def test_history_builtin_tracks_repl_commands(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(stdout=stdout, stderr=stderr, interactive=True)
    events = iter([
        "echo one",
        "echo two",
        "history 2",
        "history -d 1",
        "history",
        "history -c",
        "history",
        EOFError,
    ])

    def fake_input(_prompt):
        event = next(events)
        if isinstance(event, type) and issubclass(event, BaseException):
            raise event
        return event

    monkeypatch.setattr("builtins.input", fake_input)

    assert shell.repl() == 0
    output = stdout.getvalue()
    assert output.startswith("one\ntwo\n")
    assert "    2  echo two\n    3  history 2\n" in output
    assert "    1  echo two\n    2  history 2\n    3  history -d 1\n    4  history\n" in output
    assert output.endswith("    1  history\n\n")
    assert stderr.getvalue() == ""


def test_input_history_navigation_restores_current_line():
    stdout = io.StringIO()
    shell = Shell(stdout=stdout, stderr=io.StringIO())
    shell.history = ["echo one", "echo two"]
    state = LineHistoryState(index=len(shell.history))

    line, moved = shell.navigate_input_history("echo draft", state, -1)
    assert moved is True
    assert line == "echo two"

    line, moved = shell.navigate_input_history(line, state, -1)
    assert moved is True
    assert line == "echo one"

    line, moved = shell.navigate_input_history(line, state, -1)
    assert moved is False
    assert line == "echo one"

    line, moved = shell.navigate_input_history(line, state, 1)
    assert moved is True
    assert line == "echo two"

    line, moved = shell.navigate_input_history(line, state, 1)
    assert moved is True
    assert line == "echo draft"


def test_apply_history_navigation_redraws_line_and_beeps_at_edges():
    stdout = io.StringIO()
    shell = Shell(stdout=stdout, stderr=io.StringIO())
    shell.history = ["short", "much longer command"]
    state = LineHistoryState(index=len(shell.history))

    line = shell.apply_history_navigation("$ ", "draft", state, -1)
    assert line == "much longer command"
    line = shell.apply_history_navigation("$ ", line, state, -1)
    assert line == "short"
    line = shell.apply_history_navigation("$ ", line, state, -1)
    assert line == "short"

    output = stdout.getvalue()
    assert "\r$ much longer command" in output
    assert "\r$ short" in output
    assert output.endswith("\a")


def test_input_history_navigation_skips_adjacent_strip_duplicates():
    stdout = io.StringIO()
    shell = Shell(stdout=stdout, stderr=io.StringIO())
    shell.history = [
        "echo one",
        "echo two",
        "  echo two  ",
        "echo three",
        "echo three   ",
    ]
    state = LineHistoryState(index=len(shell.history))

    line = shell.apply_history_navigation("$ ", "draft", state, -1)
    assert line == "echo three"
    assert state.index == 3

    line = shell.apply_history_navigation("$ ", line, state, -1)
    assert line == "echo two"
    assert state.index == 1

    line = shell.apply_history_navigation("$ ", line, state, -1)
    assert line == "echo one"
    assert state.index == 0

    line = shell.apply_history_navigation("$ ", line, state, -1)
    assert line == "echo one"
    assert state.index == 0

    line = shell.apply_history_navigation("$ ", line, state, 1)
    assert line == "  echo two  "
    assert state.index == 2

    line = shell.apply_history_navigation("$ ", line, state, 1)
    assert line == "echo three   "
    assert state.index == 4

    line = shell.apply_history_navigation("$ ", line, state, 1)
    assert line == "draft"
    assert state.index == len(shell.history)

    output = stdout.getvalue()
    assert output.count("\a") == 1


def test_input_backspace_erases_wide_character_cells():
    stdout = io.StringIO()
    shell = Shell(stdout=stdout, stderr=io.StringIO())

    assert shell.erase_input_character("a中") == "a"
    assert stdout.getvalue() == "\b\b  \b\b"


def test_input_redraw_uses_terminal_display_width_for_cjk():
    stdout = io.StringIO()
    shell = Shell(stdout=stdout, stderr=io.StringIO())

    shell.redraw_input_line("$ ", "echo 中文", "echo x")

    assert terminal_display_width("echo 中文") == 9
    assert stdout.getvalue().startswith("\r" + (" " * 11) + "\r$ echo x")


def test_tab_completion_completes_single_file_and_common_prefix(tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        (tmp_path / "alpha.txt").write_text("", encoding="utf-8")
        shell = Shell(stdout=io.StringIO(), stderr=io.StringIO())
        assert shell.complete_line_for_tab("cat al").line == "cat alpha.txt"

        (tmp_path / "alpine.txt").write_text("", encoding="utf-8")
        result = shell.complete_line_for_tab("cat al")
        assert result.line == "cat alp"
        assert result.beep is False
        assert result.listings == ()
    finally:
        os.chdir(old_cwd)


def test_tab_completion_lists_matches_when_prefix_is_already_lcp(tmp_path):
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        for index in range(12):
            (tmp_path / f"ap{index:02d}.txt").write_text("", encoding="utf-8")
        stdout = io.StringIO()
        shell = Shell(stdout=stdout, stderr=io.StringIO())
        result = shell.complete_line_for_tab("echo ap")
        assert result.line == "echo ap"
        assert result.beep is True
        assert result.listings == tuple(f"ap{index:02d}.txt" for index in range(10))
        assert result.hidden_count == 2
        shell.render_completion_result(result, "$ ", "echo ap")
        rendered = stdout.getvalue()
        assert rendered.startswith("\a\nap00.txt\n")
        assert "... 2 terms hidden ...\n$ echo ap" in rendered
    finally:
        os.chdir(old_cwd)


def test_py_web_ssh_cwd_prompt_injection_without_native_utilities(monkeypatch, tmp_path):
    token = "testtoken"
    stdout = io.StringIO()
    stderr = io.StringIO()
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        (tmp_path / "visible.txt").write_text("visible\n", encoding="utf-8")
        shell = Shell(stdout=stdout, stderr=stderr, env={"PATH": ""}, interactive=True)
        commands = (
            "__py_web_ssh_cwd_armed=; "
            "__py_web_ssh_cwd_ready(){ "
            "[ -n \"${__py_web_ssh_cwd_armed-}\" ] || return; "
            "printf '\\033]6970;ready;testtoken=1\\007' >&2; "
            "}",
            "__py_web_ssh_cwd_list(){ "
            "[ -n \"${__py_web_ssh_cwd_armed-}\" ] || return; "
            "__py_web_ssh_cwd_ls=$(LC_ALL=C command ls -al 2>&1 | command base64 | command tr -d '\\r\\n') || __py_web_ssh_cwd_ls=; "
            "printf '\\033]6970;ls;testtoken=%s\\007' \"$__py_web_ssh_cwd_ls\" >&2; "
            "}",
            "__py_web_ssh_cwd_report(){ "
            "[ -n \"${__py_web_ssh_cwd_armed-}\" ] || return; "
            "__py_web_ssh_cwd_now=$(command pwd 2>/dev/null || printf '%s' \"$PWD\") || return; "
            "if [ \"${__py_web_ssh_cwd_now}\" != \"${__py_web_ssh_cwd_last-}\" ]; then "
            "__py_web_ssh_cwd_last=$__py_web_ssh_cwd_now; "
            "printf '\\033]6970;cwd;testtoken=%s\\007' \"$__py_web_ssh_cwd_now\" >&2; "
            "__py_web_ssh_cwd_list; "
            "fi; "
            "__py_web_ssh_cwd_ready; "
            "}",
            "if [ -n \"${BASH_VERSION:-}\" ]; then "
            "PROMPT_COMMAND=\"__py_web_ssh_cwd_report${PROMPT_COMMAND:+;$PROMPT_COMMAND}\"; "
            "elif [ -n \"${ZSH_VERSION:-}\" ]; then "
            "autoload -Uz add-zsh-hook 2>/dev/null && add-zsh-hook precmd __py_web_ssh_cwd_report || "
            "precmd_functions+=(__py_web_ssh_cwd_report); "
            "else "
            "__py_web_ssh_cwd_prompt=; "
            "PS1='${__py_web_ssh_cwd_prompt:-$(__py_web_ssh_cwd_report)}'\"${PS1-}\"; "
            "PS2='${__py_web_ssh_cwd_prompt:-$(__py_web_ssh_cwd_report)}'\"${PS2-}\"; "
            "fi",
        )
        for command in commands:
            assert shell.execute(command) == 0, stderr.getvalue()
        assert shell.execute("__py_web_ssh_cwd_armed=1") == 0
        stderr.seek(0)
        stderr.truncate(0)
        prompts: list[str] = []

        def fake_input(prompt):
            prompts.append(prompt)
            raise EOFError

        monkeypatch.setattr("builtins.input", fake_input)

        assert shell.repl() == 0
        hidden = stderr.getvalue()
        assert prompts == [""]
        assert f"\x1b]6970;cwd;{token}=".encode().decode() in hidden
        assert f"\x1b]6970;ls;{token}=" in hidden
        assert f"\x1b]6970;ready;{token}=1\x07" in hidden
        listing_payload = hidden.split(f"\x1b]6970;ls;{token}=", 1)[1].split("\x07", 1)[0]
        listing_text = base64.b64decode(listing_payload).decode("utf-8", errors="replace")
        visible_line = next(line for line in listing_text.splitlines() if line.endswith(" visible.txt"))
        assert len(visible_line.split(maxsplit=8)) == 9
    finally:
        os.chdir(old_cwd)


def test_py_web_ssh_base64_transfer_commands_without_native_utilities(tmp_path):
    raw = b"\x00hello\r\nweb-ssh\xff"
    source = tmp_path / "source.bin"
    encoded = tmp_path / "source.b64"
    decoded_d = tmp_path / "decoded-d.bin"
    decoded_bsd = tmp_path / "decoded-bsd.bin"
    size_path = tmp_path / "size.txt"
    temp = tmp_path / "upload.tmp"
    final = tmp_path / "final.bin"
    b64_temp = tmp_path / "upload.b64"
    err = tmp_path / "upload.tmp.err"
    source.write_bytes(raw)
    b64_temp.write_bytes(base64.b64encode(raw))
    script = f'''
command base64 < "{source}" | command tr -d '\\r\\n' > "{encoded}"
command base64 -d < "{encoded}" > "{decoded_d}"
command base64 -D < "{encoded}" > "{decoded_bsd}"
if [ -f "{source}" ]; then wc -c < "{source}" > "{size_path}"; else exit 2; fi
set -e
rm -f "{temp}" "{err}"
if command base64 -d < "{b64_temp}" > "{temp}" 2> "{err}"; then
  :
elif command base64 -D < "{b64_temp}" > "{temp}" 2> "{err}"; then
  :
else
  cat "{err}" >&2
  exit 1
fi
mv -f "{temp}" "{final}"
rm -f "{b64_temp}" "{err}"
'''
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})
    assert status == 0
    assert stdout == ""
    assert stderr == ""
    assert base64.b64decode(encoded.read_bytes()) == raw
    assert decoded_d.read_bytes() == raw
    assert decoded_bsd.read_bytes() == raw
    assert int(size_path.read_text(encoding="utf-8").strip()) == len(raw)
    assert final.read_bytes() == raw
    assert not b64_temp.exists()
    assert not err.exists()


def test_internal_extended_text_utilities_without_path(tmp_path):
    data = tmp_path / "data.txt"
    csv = tmp_path / "data.csv"
    data.write_text("beta\nalpha\nalpha\ngamma\n", encoding="utf-8")
    csv.write_text("1,red\n2,blue\n", encoding="utf-8")
    script = f'''
sort "{data}" | uniq
sort -r "{data}" | head -n 1
cut -d, -f2 "{csv}"
grep -c alpha "{data}"
egrep "alpha|gamma" "{data}" | wc -l
fgrep "a.b" "{data}"
echo fixed-miss:$?
sed -n "s/alpha/ALPHA/p" "{data}"
gawk -F, '{{ print $2 }}' "{csv}"
yes ok | head -n 2
'''
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})
    assert status == 0
    assert "alpha\nbeta\ngamma\n" in stdout
    assert "gamma\n" in stdout
    assert "red\nblue\n" in stdout
    assert "\n2\n" in stdout
    assert "fixed-miss:1\n" in stdout
    assert "ALPHA\nALPHA\n" in stdout
    assert stdout.endswith("ok\nok\n")
    assert stderr == ""


def test_package_script_posix_idioms_supported_without_path(tmp_path):
    script_path = tmp_path / "package.sh"
    seen = tmp_path / "seen.txt"
    script = f'''
set -eu
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
printf 'script:%s\\n' "$script_dir"
printf 'foo\\nfoobar\\n' > "{seen}"
grep -Fqx foo "{seen}"
printf 'grep-exact:%s\\n' "$?"
if grep -Fqx oo "{seen}"; then
  printf 'grep-partial:bad\\n'
else
  printf 'grep-partial:1\\n'
fi
printf 'DLL Name: foo.dll\\n' | sed -n 's/.*DLL Name:[[:space:]]*//p'
printf 'x => /tmp/libx.so (0x1)\\n' | sed -n -e 's/.*=>[[:space:]]*\\(\\/[^ ]*\\).*/\\1/p'
printf '  keep  spaces  \\n' | while IFS= read -r dep_spec; do printf '<%s>\\n' "$dep_spec"; done
printf 'base:%s\\n' "$(basename -- "$0")"
printf 'dir:%s\\n' "$(dirname -- "$0")"
'''
    script_path.write_text(script, encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()
    shell = Shell(argv0=str(script_path), stdout=stdout, stderr=stderr, env={"PATH": ""})

    status = shell.execute(script)

    assert status == 0
    lines = stdout.getvalue().splitlines()
    assert os.path.normcase(lines[0]) == os.path.normcase(f"script:{tmp_path}")
    assert lines[1:] == [
        "grep-exact:0",
        "grep-partial:1",
        "foo.dll",
        "/tmp/libx.so",
        "<  keep  spaces  >",
        "base:package.sh",
        f"dir:{tmp_path}",
    ]
    assert stderr.getvalue() == ""


def test_internal_clear_utility_without_path():
    status, stdout, stderr, _shell = run_shell("clear", env={"PATH": ""})
    assert status == 0
    assert stdout == "\033[H\033[2J"
    assert stderr == ""


def test_internal_df_and_lsblk_storage_fallback(monkeypatch):
    gib = 1024**3
    fake_volume = StorageVolume(
        device="C:",
        mountpoint="C:\\",
        fstype="NTFS",
        label="System",
        total=100 * gib,
        used=40 * gib,
        free=60 * gib,
        drive_type="fixed",
        serial="ABCD1234",
    )

    def fake_collect(paths=None):
        assert paths in (None, ["C:/"])
        return [fake_volume]

    monkeypatch.setattr(posix_utils, "collect_storage_volumes", fake_collect)

    status, stdout, stderr, _shell = run_shell(
        "df -h; df -T C:/; lsblk; lsblk -f; lsblk -b -o NAME,SIZE,FSAVAIL,FSUSE%,MOUNTPOINTS",
        env={"PATH": ""},
    )

    assert status == 0
    assert "Filesystem Size Used Avail Use% Mounted on\n" in stdout
    assert "C:" in stdout
    assert "100G  40G   60G  40% C:\\" in stdout
    assert "Filesystem Type 1K-blocks Used     Available Use% Mounted on\n" in stdout
    assert "104857600" in stdout
    assert "41943040" in stdout
    assert "62914560" in stdout
    assert "NAME" in stdout
    assert "MAJ:MIN" in stdout
    assert "MOUNTPOINTS" in stdout
    assert "sda" in stdout
    assert "\u2514\u2500sda1" in stdout
    assert "NTFS" in stdout
    assert "System" in stdout
    assert "ABCD1234" in stdout
    assert "107374182400" in stdout
    assert "64424509440" in stdout
    assert stderr == ""


def test_internal_help_utility_without_path():
    status, stdout, stderr, _shell = run_shell(
        "help; help cd; help clear; help nope; echo status:$?; command -v help; type help",
        env={"PATH": ""},
    )
    assert status == 0
    assert "py-posix-shell, version" in stdout
    assert "These shell commands and fallback utilities are defined internally." in stdout
    assert "if COMMANDS; then COMMANDS;" in stdout
    assert "help [name ...]" in stdout
    assert "history [-c] [-d offset] [n]" in stdout
    assert "ps [aux|-ef]" in stdout
    assert "top [-b] [-n count]" in stdout
    assert "which [-a] name ..." in stdout
    assert "cd [dir]\n    Change the current directory." in stdout
    assert "clear\n    Clear the terminal using an ANSI fallback sequence." in stdout
    assert "status:1\n" in stdout
    assert "\nhelp\n" in stdout
    assert "help is a shell utility" in stdout
    assert "help: no help topics match 'nope'\n" == stderr


def test_help_prefers_internal_utility_even_when_external_exists(monkeypatch):
    shell = Shell(stdout=io.StringIO(), stderr=io.StringIO(), env={"PATH": "C:\\fake"})

    def fake_resolve(name, _env):
        if name == "help":
            return "C:\\Windows\\System32\\help.exe"
        return None

    monkeypatch.setattr(shell, "resolve_command", fake_resolve)

    status = shell.execute("help cd; command -v help; command -V help; type help")
    stdout = shell.stdout.getvalue()

    assert status == 0
    assert "cd [dir]\n    Change the current directory." in stdout
    assert "\nhelp\n" in stdout
    assert "help is a shell utility\n" in stdout
    assert "C:\\Windows\\System32\\help.exe" not in stdout
    assert shell.stderr.getvalue() == ""


def test_windows_vi_fallback_without_path(tmp_path):
    if os.name != "nt":
        return
    target = tmp_path / "note.txt"
    script = f'printf "i\\nhello\\nworld\\n.\\nwq\\n" | vi "{target}"'
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})
    assert status == 2
    assert stdout == ""
    assert "fallback editor requires a TTY" in stderr
    assert not target.exists()


def test_windows_vi_editor_core_writes_file(tmp_path):
    target = tmp_path / "note.txt"
    editor = WindowsViEditor(target, io.StringIO(), io.StringIO())
    editor.handle_normal_key("i")
    for char in "hello":
        editor.handle_insert_key(char)
    editor.handle_insert_key("ENTER")
    for char in "world":
        editor.handle_insert_key(char)
    editor.handle_insert_key("ESC")
    editor.execute_command("wq")
    assert editor.exit_status == 0
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"


def test_windows_vi_editor_insert_mode_arrows(tmp_path):
    target = tmp_path / "note.txt"
    editor = WindowsViEditor(target, io.StringIO(), io.StringIO())
    editor.handle_normal_key("i")
    for char in "abc":
        editor.handle_insert_key(char)
    editor.handle_insert_key("LEFT")
    editor.handle_insert_key("X")
    editor.execute_command("wq")
    assert target.read_text(encoding="utf-8") == "abXc\n"


def test_windows_vi_editor_render_shows_cursor_and_scrolls_horizontally(monkeypatch, tmp_path):
    stdout = io.StringIO()
    editor = WindowsViEditor(tmp_path / "note.txt", stdout, io.StringIO())
    editor.lines = ["0123456789abc"]
    editor.cursor_col = 12
    monkeypatch.setattr(posix_utils.shutil, "get_terminal_size", lambda fallback: os.terminal_size((10, 5)))

    editor.render()
    rendered = stdout.getvalue()

    assert "\033[?25l" in rendered
    assert rendered.endswith("\033[?25h")
    assert "3456789abc" in rendered
    assert "\033[1;10H\033[?25h" in rendered
    assert editor.left_col == 3


def test_windows_vi_editor_render_uses_visual_tab_columns(monkeypatch, tmp_path):
    stdout = io.StringIO()
    editor = WindowsViEditor(tmp_path / "note.txt", stdout, io.StringIO())
    editor.lines = ["\tab"]
    editor.cursor_col = 1
    monkeypatch.setattr(posix_utils.shutil, "get_terminal_size", lambda fallback: os.terminal_size((20, 5)))

    editor.render()

    assert "\033[1;5H\033[?25h" in stdout.getvalue()


def test_windows_vi_prefers_available_vim(monkeypatch):
    if os.name != "nt":
        return
    shell = Shell(env={"PATH": "C:\\fake"})

    def fake_which(name, path=None):
        if name == "vim":
            return "C:\\tools\\vim.exe"
        return None

    monkeypatch.setattr(shell_module.shutil, "which", fake_which)

    assert shell.resolve_command("vi", shell.env) == "C:\\tools\\vim.exe"
    assert shell.should_run_internal_utility("vi", shell.env) is False


def test_internal_findutils_and_file_install_without_path(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("payload\n", encoding="utf-8")
    installed = tmp_path / "dest" / "bin" / "source.txt"
    database = tmp_path / "locatedb"
    script = f'''
install -D -m 755 "{source}" "{installed}"
cat "{installed}"
find "{tmp_path}" -name "*.txt" | sort
printf "one two\\n" | xargs -n1 echo item
updatedb -o "{database}" -U "{tmp_path}"
locate -d "{database}" source.txt
unlink "{installed}"
'''
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})
    assert status == 0
    assert "payload\n" in stdout
    assert str(source) in stdout
    assert str(installed) in stdout
    assert "item one\nitem two\n" in stdout
    assert not installed.exists()
    assert stderr == ""


def test_internal_find_supports_common_gnu_style_expressions(tmp_path):
    root = tmp_path / "tree"
    src = root / "src"
    build = root / "build"
    src.mkdir(parents=True)
    build.mkdir()
    (src / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (src / "README.MD").write_text("readme\n", encoding="utf-8")
    (src / "notes.txt").write_text("notes\n", encoding="utf-8")
    (build / "generated.py").write_text("generated\n", encoding="utf-8")
    (root / "empty").mkdir()
    (root / "zero.txt").write_text("", encoding="utf-8")

    status, stdout, stderr, _shell = run_shell(
        f'''
find "{root}" -maxdepth 2 \\( -name "*.py" -o -iname "readme.*" \\) -type f | sort
printf "\\n--dirs--\\n"
find "{root}" -mindepth 1 -maxdepth 1 -type d | sort
printf "\\n--not-py--\\n"
find "{root}" ! -name "*.py" -type f | sort
printf "\\n--exec--\\n"
find "{root}" -path "*src*" -type f -exec basename {{}} \\; | sort
printf "\\n--empty0--\\n"
find "{root}" -empty -print0
''',
        env={"PATH": ""},
    )

    assert status == 0
    assert str(src / "README.MD") in stdout
    assert str(src / "app.py") in stdout
    assert str(build / "generated.py") in stdout
    assert f"--dirs--\n{build}\n{root / 'empty'}\n{src}\n" in stdout
    assert str(src / "notes.txt") in stdout
    assert str(root / "zero.txt") in stdout
    assert "--exec--\nREADME.MD\napp.py\nnotes.txt\n" in stdout
    assert str(root / "empty") + "\0" in stdout
    assert str(root / "zero.txt") + "\0" in stdout
    assert stderr == ""


def test_find_current_directory_keeps_dot_prefix(tmp_path):
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        status, stdout, stderr, _shell = run_shell("find . -maxdepth 1 -type f", env={"PATH": ""})
    finally:
        os.chdir(old_cwd)

    assert status == 0
    assert stdout == "." + os.sep + "one.txt\n"
    assert stderr == ""


def test_windows_find_internal_fallback_replaces_system_find(monkeypatch):
    monkeypatch.setattr(shell_module.os, "name", "nt")
    shell = Shell(env={"PATH": r"C:\Windows\System32", "PATHEXT": ".EXE"})

    assert shell.should_run_internal_utility("find", shell.env) is True


def test_internal_diffutils_and_tar_without_path(tmp_path):
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    base = tmp_path / "base.txt"
    tar_src = tmp_path / "tar-src"
    tar_out = tmp_path / "tar-out"
    archive = tmp_path / "archive.tar"
    left.write_text("one\ntwo\n", encoding="utf-8")
    right.write_text("one\nthree\n", encoding="utf-8")
    base.write_text("one\n", encoding="utf-8")
    tar_src.mkdir()
    (tar_src / "entry.txt").write_text("inside\n", encoding="utf-8")
    tar_out.mkdir()
    script = f'''
diff -u "{left}" "{right}"
echo diff:$?
cmp -s "{left}" "{left}"
cmp -s "{left}" "{right}"
echo cmp:$?
diff3 "{left}" "{base}" "{right}"
echo diff3:$?
sdiff "{left}" "{right}"
echo sdiff:$?
tar -cf "{archive}" -C "{tar_src}" entry.txt
tar -tf "{archive}"
tar -xf "{archive}" -C "{tar_out}"
cat "{tar_out / 'entry.txt'}"
'''
    status, stdout, stderr, _shell = run_shell(script, env={"PATH": ""})
    assert status == 0
    assert "--- " in stdout and "+++ " in stdout
    assert "diff:1\n" in stdout
    assert "cmp:1\n" in stdout
    assert "diff3:1\n" in stdout
    assert "sdiff:1\n" in stdout
    assert "entry.txt\ninside\n" in stdout
    assert stderr == ""
