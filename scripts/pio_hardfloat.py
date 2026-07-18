Import("env")

# Ensure final ELF link uses hard-float ABI to match compiled objects/libraries.
env.Append(
    LINKFLAGS=[
        "-mfpu=fpv4-sp-d16",
        "-mfloat-abi=hard",
    ]
)
