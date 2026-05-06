# Anti-cheat fixtures

`benchmark/tests` 使用的反作弊样例代码，与 pytest 逻辑解耦，方便补充
机械层检查覆盖。

## 机械层 fixture (`<case>/relu_impl.py`)

| 目录          | 形态                                | cheat_detector 期望 verdict |
|---------------|-------------------------------------|------------------------------|
| `clean/`      | 1 个 `@pypto.jit`, body 真用 pypto  | `pass`                       |
| `no_pypto/`   | 不 `import pypto`, 纯 `torch.relu`  | `cheat` (import_pypto+has_jit)|
| `no_jit/`     | `import pypto` 但 0 个 `@pypto.jit` | `cheat` (has_jit)            |
| `suspicious/` | 1 个 jit + "workaround" 注释        | `suspicious` (forbidden_text)|
