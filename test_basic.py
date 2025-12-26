"""
基础测试脚本
验证各个模块的基本功能
"""
import asyncio
import sys


async def test_config():
    """测试配置模块"""
    print("测试配置模块...")
    try:
        from config import read_global_config
        config = await read_global_config()
        print(f"✓ 配置加载成功")
        print(f"  - API Endpoint: {config.api_endpoint}")
        print(f"  - Port: {config.port}")
        print(f"  - Has Refresh Token: {bool(config.refresh_token)}")
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


async def test_models():
    """测试数据模型"""
    print("\n测试数据模型...")
    try:
        from models import ClaudeRequest, ClaudeMessage, ClaudeTool

        # 创建测试请求
        message = ClaudeMessage(role="user", content="Hello")
        request = ClaudeRequest(
            model="claude-sonnet-4.5",
            messages=[message],
            max_tokens=1024
        )

        print(f"✓ 数据模型创建成功")
        print(f"  - Model: {request.model}")
        print(f"  - Messages: {len(request.messages)}")
        return True
    except Exception as e:
        print(f"✗ 数据模型测试失败: {e}")
        return False


async def test_converter():
    """测试请求转换"""
    print("\n测试请求转换...")
    try:
        from models import ClaudeRequest, ClaudeMessage
        from converter import convert_claude_to_codewhisperer_request

        # 创建测试请求
        message = ClaudeMessage(role="user", content="Hello")
        claude_req = ClaudeRequest(
            model="claude-sonnet-4.5",
            messages=[message],
            max_tokens=1024
        )

        # 转换请求
        cw_req = convert_claude_to_codewhisperer_request(claude_req)

        print(f"✓ 请求转换成功")
        print(f"  - Conversation ID: {cw_req.conversationState.conversationId}")
        print(f"  - Model ID: {cw_req.conversationState.currentMessage.userInputMessage.modelId}")
        return True
    except Exception as e:
        print(f"✗ 请求转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_parser():
    """测试事件解析"""
    print("\n测试事件解析...")
    try:
        from parser import parse_event_data, parse_sse_line

        # 测试 SSE 行解析
        line = "data: {\"type\": \"message_start\"}"
        data = parse_sse_line(line)
        assert data == '{"type": "message_start"}', "SSE 行解析失败"

        # 测试事件解析
        event_json = '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}'
        event = parse_event_data(event_json)
        assert event is not None, "事件解析失败"

        print(f"✓ 事件解析成功")
        print(f"  - Event Type: {type(event).__name__}")
        return True
    except Exception as e:
        print(f"✗ 事件解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_opus_model_mapping():
    """测试 Opus 模型映射"""
    print("\n测试 Opus 模型映射...")
    try:
        from converter import map_claude_model_to_amazonq

        # 测试各种 Opus 模型名称
        test_cases = [
            ("claude-opus-4.5", "claude-opus-4.5"),
            ("claude-opus-4-5", "claude-opus-4.5"),
            ("claude-opus-4", "claude-opus-4.5"),
            ("Claude-Opus-4.5", "claude-opus-4.5"),  # 大小写测试
            ("claude-3-opus-20240229", "claude-opus-4.5"),  # Legacy
        ]

        for input_model, expected_output in test_cases:
            result = map_claude_model_to_amazonq(input_model)
            assert result == expected_output, f"映射失败: {input_model} -> {result} (期望: {expected_output})"
            print(f"  ✓ {input_model} -> {result}")

        print(f"✓ Opus 模型映射测试通过")
        return True
    except Exception as e:
        print(f"✗ Opus 模型映射测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("=" * 50)
    print("Amazon Q to Claude API Proxy - 基础测试")
    print("=" * 50)

    results = []

    # 运行测试
    results.append(await test_config())
    results.append(await test_models())
    results.append(await test_converter())
    results.append(await test_parser())
    results.append(await test_opus_model_mapping())

    # 汇总结果
    print("\n" + "=" * 50)
    print("测试结果汇总")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
