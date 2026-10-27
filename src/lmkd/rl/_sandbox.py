"""Linux child restrictions for trusted code verification; no import-time side effects."""
import ctypes
import errno
import os
import platform


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _Program(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


def _check(result):
    if result < 0:
        raise OSError(ctypes.get_errno(), "child sandbox restriction failed")
    return result


def _restrict_writes(libc, tmp):
    abi = _check(libc.syscall(444, 0, 0, 1))
    if abi < 3:
        raise OSError("Landlock ABI >= 3 is required to restrict truncate operations")
    # Landlock write/remove/create/refer/truncate rights. Read and execute remain available.
    access = sum(1 << bit for bit in [1, *range(4, 13)])
    if abi >= 2:
        access |= 1 << 13
    if abi >= 3:
        access |= 1 << 14
    attr = _Ruleset(access)
    ruleset = _check(libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0))
    directory = os.open(tmp, os.O_PATH | os.O_CLOEXEC)
    try:
        path = _PathBeneath(access, directory)
        _check(libc.syscall(445, ruleset, 1, ctypes.byref(path), 0))
        _check(libc.syscall(446, ruleset, 0))
    finally:
        os.close(directory)
        os.close(ruleset)


def _restrict_network(libc):
    # x86-64 audit arch check; reject x32 ABI and socket/io_uring calls (including alternate networking paths).
    instructions = [_Filter(0x20, 0, 0, 4), _Filter(0x15, 1, 0, 0xC000003E),
                    _Filter(0x06, 0, 0, 0x80000000), _Filter(0x20, 0, 0, 0),
                    _Filter(0x35, 0, 1, 0x40000000), _Filter(0x06, 0, 0, 0x00050000 | errno.EPERM)]
    for number in [41, 42, 43, 44, 45, 46, 47, 49, 50, 53, 288, 299, 307, 425, 426, 427]:
        instructions += [_Filter(0x15, 0, 1, number), _Filter(0x06, 0, 0, 0x00050000 | errno.EPERM)]
    instructions.append(_Filter(0x06, 0, 0, 0x7FFF0000))
    array = (_Filter * len(instructions))(*instructions)
    program = _Program(len(instructions), array)
    _check(libc.prctl(22, 2, ctypes.byref(program), 0, 0))


def restrict_child(tmp):
    """Fail closed unless Linux x86-64 can deny sockets and writes outside tmp."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise OSError("code verifier requires Linux x86-64 Landlock/seccomp")
    libc = ctypes.CDLL(None, use_errno=True)
    _check(libc.prctl(38, 1, 0, 0, 0))  # no_new_privs, inherited through exec
    _restrict_writes(libc, tmp)
    _restrict_network(libc)


def _main():
    import resource
    import runpy
    import sys

    tmp, target, memory_mb = sys.argv[1:]
    memory = int(memory_mb) * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    restrict_child(tmp)
    sys.argv = [target]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    _main()
