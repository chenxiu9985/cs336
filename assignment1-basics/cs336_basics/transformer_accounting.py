def resource_accounting(
    vocab_size,
    context_length,
    num_layers,
    d_model,
    num_heads,
    d_ff,
):
    assert d_model % num_heads == 0

    V = vocab_size
    S = context_length
    L = num_layers
    D = d_model
    F = d_ff

    # ---------- 参数量 ----------
    params = {
        "Token embedding": V * D,
        "Attention projections": L * 4 * D * D,
        "FFN": L * 3 * D * F,
        "Block RMSNorm": L * 2 * D,
        "Final RMSNorm": D,
        "LM head": D * V,
    }

    # ---------- 矩阵乘法 FLOPs ----------
    # 所有 Block 的计算量已乘以 L。
    # 注意力按完整 S × S 矩阵计算，与当前实现一致。
    flops = {
        # Q、K、V：3 次 (S, D) @ (D, D)
        "Q/K/V projections": L * 3 * 2 * S * D * D,

        # 所有 heads 合计：
        # num_heads * 2 * S * S * (D / num_heads)
        "Attention QK^T": L * 2 * S * S * D,

        # attention_weights @ V
        "Attention AV": L * 2 * S * S * D,

        # (S, D) @ (D, D)
        "Attention output projection": L * 2 * S * D * D,

        # w1、w3：(S, D) @ (D, F)
        # w2：    (S, F) @ (F, D)
        "FFN": L * 3 * 2 * S * D * F,

        # (S, D) @ (D, V)
        "LM head": 2 * S * D * V,
    }

    return {
        "params": params,
        "flops": flops,
        "total_params": sum(params.values()),
        "total_flops": sum(flops.values()),
    }


def print_report(name, result):
    total_params = result["total_params"]
    total_flops = result["total_flops"]
    memory_bytes = total_params * 4  # FP32

    print(f"\n{name}")
    print(f"Parameters: {total_params:,}")
    print(
        f"FP32 parameter memory: "
        f"{memory_bytes / 10**9:.4f} GB "
        f"({memory_bytes / 2**30:.4f} GiB)"
    )

    print("\nParameter breakdown:")
    for component, count in result["params"].items():
        print(f"  {component:<30} {count:>18,}")

    print("\nMatrix multiplication FLOPs:")
    for component, count in result["flops"].items():
        percentage = count / total_flops * 100
        print(
            f"  {component:<30} "
            f"{count:>20,}  "
            f"{percentage:>6.2f}%"
        )

    print(f"Total: {total_flops:,} FLOPs")
    print(f"       {total_flops / 10**12:.6f} TFLOPs")

    largest = max(result["flops"], key=result["flops"].get)
    print(f"Largest component: {largest}")


def nearest_d_ff(d_model):
    # 按截图中的规则：
    # 取最接近 (8/3) * d_model 的 64 的倍数。
    return 64 * round(((8 / 3) * d_model) / 64)


def main():
    vocab_size = 50_257

    # (a)、(b)、(c)：XL 模型
    xl_config = dict(
        vocab_size=vocab_size,
        context_length=1_024,
        num_layers=48,
        d_model=1_600,
        num_heads=25,
        d_ff=4_288,
    )

    xl = resource_accounting(**xl_config)
    print_report("GPT-2 XL-shaped assignment model", xl)

    # (d)：其他规模，保持词表大小和上下文长度相同。
    # 假设 d_ff 延续题目中“最接近的 64 的倍数”规则。
    for name, layers, d_model, heads in [
        ("Small", 12, 768, 12),
        ("Medium", 24, 1_024, 16),
        ("Large", 36, 1_280, 20),
    ]:
        d_ff = nearest_d_ff(d_model)

        result = resource_accounting(
            vocab_size=vocab_size,
            context_length=1_024,
            num_layers=layers,
            d_model=d_model,
            num_heads=heads,
            d_ff=d_ff,
        )
        print_report(f"{name} (d_ff={d_ff})", result)

    # (e)：XL 上下文长度从 1,024 增加到 16,384
    long_config = {
        **xl_config,
        "context_length": 16_384,
    }
    xl_long = resource_accounting(**long_config)
    print_report("XL with context length 16,384", xl_long)

    print("\nContext length comparison:")
    print(
        "Total FLOPs multiplier: "
        f"{xl_long['total_flops'] / xl['total_flops']:.4f}x"
    )

    for component, old_count in xl["flops"].items():
        new_count = xl_long["flops"][component]
        old_share = old_count / xl["total_flops"] * 100
        new_share = new_count / xl_long["total_flops"] * 100

        print(
            f"  {component:<30} "
            f"{new_count / old_count:>6.1f}x  "
            f"share: {old_share:.2f}% -> {new_share:.2f}%"
        )


if __name__ == "__main__":
    main()