"""Tests for command_precheck module."""
import shutil
from unittest.mock import patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from command_precheck import precheck_command, PrecheckResult, COMMAND_FALLBACK_MAP


def _mock_which(cmd):
    """Mock shutil.which: 模拟容器环境中只有基础命令"""
    available = {
        "grep", "find", "cat", "ls", "sed", "awk", "cd", "echo",
        "python3", "pip", "curl", "wget", "diff", "du", "ps",
        "wc", "time", "head", "tail", "sort", "uniq", "nl",
    }
    return f"/usr/bin/{cmd}" if cmd in available else None


class TestPrecheckSimpleCommands:
    """测试简单命令的预检和降级"""
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_available_command_unchanged(self, mock_w):
        result = precheck_command("grep -rn 'hello' /app")
        assert result.rewritten_command == "grep -rn 'hello' /app"
        assert result.substitutions == []
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_replaced_by_grep(self, mock_w):
        result = precheck_command("rg -n 'pattern' /app")
        assert "grep" in result.rewritten_command
        assert len(result.substitutions) == 1
        assert result.substitutions[0]["original"] == "rg"
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_fd_replaced_by_find(self, mock_w):
        result = precheck_command("fd '*.py'")
        assert "find" in result.rewritten_command
        assert len(result.substitutions) == 1
        assert result.substitutions[0]["original"] == "fd"
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_bat_replaced_by_cat(self, mock_w):
        result = precheck_command("bat main.py")
        assert "cat" in result.rewritten_command
        assert result.substitutions[0]["original"] == "bat"
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_unknown_missing_command(self, mock_w):
        result = precheck_command("nonexistent_tool --arg")
        assert result.rewritten_command == "nonexistent_tool --arg"
        assert result.substitutions == []
        assert "nonexistent_tool" in result.missing_commands


class TestPrecheckCompoundCommands:
    """测试复合命令（&&, ||, |, ;）的预检和降级"""
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_cd_and_rg(self, mock_w):
        """cd /app && rg pattern → cd /app && grep -rn pattern"""
        result = precheck_command("cd /app && rg -n 'pattern' .")
        assert "grep" in result.rewritten_command
        assert "cd /app" in result.rewritten_command
        assert len(result.substitutions) == 1
        assert result.substitutions[0]["original"] == "rg"
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_pipe_with_missing_first(self, mock_w):
        """rg pattern | head → grep -rn pattern | head"""
        result = precheck_command("rg 'pattern' /app | head -20")
        assert "grep" in result.rewritten_command
        assert "head" in result.rewritten_command
        assert len(result.substitutions) == 1
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_multiple_replacements(self, mock_w):
        """rg pattern && bat file → grep -rn pattern && cat file"""
        result = precheck_command("rg 'x' /app && bat output.txt")
        assert "grep" in result.rewritten_command
        assert "cat" in result.rewritten_command
        assert len(result.substitutions) == 2
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_semicolon_separator(self, mock_w):
        """rg foo; bat bar → grep -rn foo; cat bar"""
        result = precheck_command("rg foo; bat bar")
        assert "grep" in result.rewritten_command
        assert "cat" in result.rewritten_command
        assert len(result.substitutions) == 2
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_or_separator(self, mock_w):
        """rg foo || echo not_found"""
        result = precheck_command("rg foo || echo not_found")
        assert "grep" in result.rewritten_command
        assert "echo not_found" in result.rewritten_command
        assert len(result.substitutions) == 1


class TestPrecheckEdgeCases:
    """测试边界情况"""
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_empty_command(self, mock_w):
        result = precheck_command("")
        assert result.rewritten_command == ""
        assert result.substitutions == []
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_builtin_command_not_checked(self, mock_w):
        """shell 内建命令不做检查"""
        result = precheck_command("cd /app")
        assert result.rewritten_command == "cd /app"
        assert result.substitutions == []
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_path_command_not_checked(self, mock_w):
        """带路径的命令不做检查"""
        result = precheck_command("/usr/local/bin/custom_tool --arg")
        assert result.rewritten_command == "/usr/local/bin/custom_tool --arg"
        assert result.substitutions == []
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_dot_prefix_not_checked(self, mock_w):
        """./script.sh 不做检查"""
        result = precheck_command("./run_test.sh")
        assert result.rewritten_command == "./run_test.sh"
        assert result.substitutions == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_env_var_prefix_preserved(self, mock_w):
        """FOO=bar rg pattern → FOO=bar grep -rn pattern"""
        result = precheck_command("FOO=bar rg 'pattern'")
        assert "grep" in result.rewritten_command
        assert "FOO=bar" in result.rewritten_command
        assert len(result.substitutions) == 1
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_extra_aliases_override(self, mock_w):
        """extra_aliases 可以覆盖默认映射"""
        result = precheck_command("rg pattern", extra_aliases={"rg": "grep -E"})
        assert "grep -E" in result.rewritten_command
        assert result.substitutions[0]["replacement"] == "grep -E"


class TestArgumentTransform:
    """测试参数转换逻辑"""
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_files_mode_to_find(self, mock_w):
        """rg --files -g '*.md' → find . -name '*.md'"""
        result = precheck_command("rg --files -g '*.md'")
        assert "find ." in result.rewritten_command
        assert "-name" in result.rewritten_command
        assert "--files" not in result.rewritten_command
        assert "grep" not in result.rewritten_command
        assert len(result.substitutions) == 1
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_files_plain(self, mock_w):
        """rg --files → find . (无参数)"""
        result = precheck_command("rg --files")
        assert "find ." in result.rewritten_command
        assert "--files" not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_files_with_hidden(self, mock_w):
        """rg --files --hidden -g '*.py' → find . -name '*.py' (drop --hidden)"""
        result = precheck_command("rg --files --hidden -g '*.py'")
        assert "find ." in result.rewritten_command
        assert "--hidden" not in result.rewritten_command
        assert "-name" in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_files_with_matches(self, mock_w):
        """rg --files-with-matches pattern → grep -rln pattern"""
        result = precheck_command("rg --files-with-matches 'hello' /app")
        assert "grep -rln" in result.rewritten_command
        assert "--files-with-matches" not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_l_flag(self, mock_w):
        """rg -l pattern → grep -rln pattern"""
        result = precheck_command("rg -l 'hello' /app")
        assert "grep -rln" in result.rewritten_command
        assert " -l " not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_normal_drops_incompatible_flags(self, mock_w):
        """rg --no-heading --smart-case pattern → grep -rn pattern (drop rg-only flags)"""
        result = precheck_command("rg --no-heading --smart-case 'pattern' /app")
        assert "grep -rn" in result.rewritten_command
        assert "--no-heading" not in result.rewritten_command
        assert "--smart-case" not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_translate_ignore_case(self, mock_w):
        """rg --ignore-case pattern → grep -rn -i pattern"""
        result = precheck_command("rg --ignore-case 'pattern' /app")
        assert "grep -rn" in result.rewritten_command
        assert "-i" in result.rewritten_command
        assert "--ignore-case" not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_drops_type_option(self, mock_w):
        """rg -t py pattern → grep -rn pattern (drop -t py)"""
        result = precheck_command("rg -t py 'pattern' /app")
        assert "grep -rn" in result.rewritten_command
        assert " -t " not in result.rewritten_command
        assert " py " not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_compound_with_files_flag(self, mock_w):
        """cd /app && rg --files -g '*.md' → cd /app && find . -name '*.md'"""
        result = precheck_command("cd /app && rg --files -g '*.md'")
        assert "cd /app" in result.rewritten_command
        assert "find ." in result.rewritten_command
        assert "-name" in result.rewritten_command
        assert "--files" not in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_rg_fixed_strings(self, mock_w):
        """rg -F 'literal' → grep -rn -F 'literal'"""
        result = precheck_command("rg -F 'literal.str' /app")
        assert "grep -rn" in result.rewritten_command
        assert "-F" in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_fd_drops_incompatible(self, mock_w):
        """fd --hidden -e py pattern → find . -name pattern (drop --hidden, -e py)"""
        result = precheck_command("fd --hidden -e py 'test'")
        assert "find" in result.rewritten_command
        assert "--hidden" not in result.rewritten_command
        # -e is drop_options for fd, should be removed along with its value
        assert " -e " not in result.rewritten_command


class TestQuoteAwareSplit:
    """测试引号感知的命令分割"""
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_pipe_inside_single_quotes_not_split(self, mock_w):
        """grep 'a|b' file 不应被拆分为 grep 'a 和 b' file"""
        result = precheck_command("grep 'a|b' /app/test.py")
        assert result.rewritten_command == "grep 'a|b' /app/test.py"
        assert result.substitutions == []
        assert result.missing_commands == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_pipe_inside_double_quotes_not_split(self, mock_w):
        """grep \"a|b\" file 不应被拆分"""
        result = precheck_command('grep "a|b" /app/test.py')
        assert 'a|b' in result.rewritten_command
        assert result.substitutions == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_or_inside_quotes_not_split(self, mock_w):
        """grep -E 'foo||bar' 不应被当成 || 分隔"""
        result = precheck_command("grep -E 'foo||bar' /app")
        assert "foo||bar" in result.rewritten_command
        assert result.substitutions == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_and_inside_quotes_not_split(self, mock_w):
        """echo 'a && b' 不应被拆分"""
        result = precheck_command("echo 'a && b'")
        assert "a && b" in result.rewritten_command
        assert result.substitutions == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_dollar_paren_subshell_not_split(self, mock_w):
        """echo $(echo a | sort) 不应拆分子 shell 内的管道"""
        result = precheck_command("echo $(echo a | sort)")
        assert "$(echo a | sort)" in result.rewritten_command
        assert result.substitutions == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_backtick_subshell_not_split(self, mock_w):
        """echo `echo a | sort` 不应拆分反引号内的管道"""
        result = precheck_command("echo `echo a | sort`")
        assert "`echo a | sort`" in result.rewritten_command
        assert result.substitutions == []
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_real_pipe_outside_quotes_still_splits(self, mock_w):
        """grep 'pattern' file | head 仍然正常拆分管道"""
        result = precheck_command("grep 'pattern' /app | head -5")
        # 两个子命令都应该存在
        assert "grep" in result.rewritten_command
        assert "head" in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_mixed_quoted_and_real_pipe(self, mock_w):
        """grep -E 'a|b' file | head 应只在真正管道处分割"""
        result = precheck_command("grep -E 'a|b' /app | head -5")
        assert "a|b" in result.rewritten_command
        assert "head" in result.rewritten_command


class TestArgumentTransformEdgeCases:
    """参数转换边界情况"""
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_positional_args_preserved(self, mock_w):
        """rg pattern path → grep -rn pattern path"""
        result = precheck_command("rg 'hello' /app/src")
        assert "grep -rn" in result.rewritten_command
        # 位置参数被保留
        assert "hello" in result.rewritten_command
        assert "/app/src" in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_no_transform_rules_passthrough(self, mock_w):
        """对没有参数转换规则的命令，参数原样保留"""
        result = precheck_command("bat --line-range 1:10 main.py")
        assert "cat" in result.rewritten_command
        # bat 没有参数转换规则，参数原样保留
        assert "--line-range" in result.rewritten_command
        assert "main.py" in result.rewritten_command
    
    @patch("command_precheck.shutil.which", side_effect=_mock_which)
    def test_available_command_no_transform(self, mock_w):
        """已存在的命令不做任何转换"""
        result = precheck_command("grep --no-heading 'test' /app")
        # grep 存在，即使有奇怪参数也不处理
        assert result.rewritten_command == "grep --no-heading 'test' /app"
        assert result.substitutions == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
