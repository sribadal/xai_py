import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

COMMANDS = {
    "export": "xai_decision_making.explainability.run_export",
    "enrich": "xai_decision_making.explainability.enrich_exports",
    "analyze": "xai_decision_making.analysis.run_all",
    "audit": "xai_decision_making.explainability.audit_co12_v2",
}

DEFAULT_ARTIFACTS = {
    "rsf": "Missingness_Avoiding_Large_(All)_paper_headline_no_leak_runnoleak_20260523_1023_fold1",
    "condor": "condor_clean28_fold5_rmse_13.88",
}


def _detect_model_type(args):
    for i, arg in enumerate(args):
        if arg == "--model-type" and i + 1 < len(args):
            return args[i + 1]
    return "rsf"


def _inject_defaults(args):
    model_type = _detect_model_type(args)
    if "--artifact" not in args:
        default = DEFAULT_ARTIFACTS.get(model_type)
        if default:
            args.extend(["--artifact", default])
    for i, arg in enumerate(args):
        if arg == "--run" and i + 1 < len(args):
            run_name = args[i + 1]
            args.pop(i)
            args.pop(i)
            out_dir = os.path.join(ROOT, "exports", run_name)
            if "--output-dir" not in args:
                args.extend(["--output-dir", out_dir])
            break
    return args


def _inject_audit_defaults(args):
    if "--rsf-artifact" not in args:
        args.extend(["--rsf-artifact", DEFAULT_ARTIFACTS["rsf"]])
    if "--condor-checkpoint" not in args:
        args.extend(["--condor-checkpoint", DEFAULT_ARTIFACTS["condor"]])
    if "--exports-dir" not in args:
        args.extend(["--exports-dir", os.path.join(ROOT, "exports")])
    return args


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        available = ", ".join(COMMANDS)
        sys.exit(f"Usage: xai <command> [args]\nCommands: {available}")
    command = sys.argv[1]
    args = sys.argv[2:]
    if command == "export":
        args = _inject_defaults(args)
    elif command == "audit":
        args = _inject_audit_defaults(args)
    sys.argv = [sys.argv[0]] + args
    module = __import__(COMMANDS[command], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()