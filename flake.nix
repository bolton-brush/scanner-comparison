{
  description = "Scanner comparison development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    pybuild.url = "github:pyproject-nix/build-system-pkgs";
    pyproject.url = "github:pyproject-nix/pyproject.nix";
    appimage.url = "github:ralismark/nix-appimage";
  };

  outputs =
    { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import inputs.nixpkgs { inherit system; };
        pkgs-treefmt = (import inputs.nixpkgs) {
          inherit system;
        };
        python = pkgs.python313;
        workspace = inputs.uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        pythonBase = pkgs.callPackage inputs.pyproject.build.packages {
          inherit python;
        };
        pythonSet = pythonBase.overrideScope (
          pkgs.lib.composeManyExtensions [
            inputs.pybuild.overlays.wheel
            overlay
          ]
        );
        venv = pythonSet.mkVirtualEnv "venv" workspace.deps.default;
        venvDev = pythonSet.mkVirtualEnv "venvDev" (workspace.deps.all or workspace.deps.default);
        inherit (pkgs.callPackages inputs.pyproject.build.util { }) mkApplication;
        treefmtconfig = inputs.treefmt-nix.lib.evalModule pkgs-treefmt {
          projectRootFile = "flake.nix";
          programs = {
            alejandra.enable = true;
            toml-sort.enable = true;
            yamlfmt.enable = true;
            mdformat = {
              enable = true;
              plugins = ps: [
                ps.mdformat-gfm
              ];
              settings = {
                wrap = 88;
                end-of-line = "lf";
              };
            };
            shellcheck.enable = true;
            shfmt.enable = true;
            nixfmt.enable = true;
          };
          settings.formatter.shellcheck.excludes = [
            ".envrc"
          ];
        };
      in
      {
        formatter = treefmtconfig.config.build.wrapper;
        devShells = {
          default = pkgs.mkShell {
            name = "scanner-comparison-env";

            buildInputs =
              with pkgs;
              [
                nil
                nixd
                uv
                ruff
                basedpyright
              ]
              ++ [
                venvDev
              ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = pythonSet.python.interpreter;
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              PROJ_ROOT=$(git rev-parse --show-toplevel)
              export PYTHONPATH="$PROJ_ROOT/src:${venvDev}/lib/*/site-packages:$PYTHONPATH"
              ln -sfn ${venvDev} $PROJ_ROOT/src/.venv
            '';
          };
        };
        packages =
          let
            app = mkApplication {
              inherit venv;
              package = pythonSet.scanner-comparison;
            };
          in
          {
            default = app;
            appimage = inputs.appimage.bundlers.${system}.default app;
            # ps = pythonSet;
            scanner-compare = app;
          };
        checks = {
          formatting = treefmtconfig.config.build.check self;
          ruff-lint = pkgs.stdenvNoCC.mkDerivation {
            name = "ruff-lint";
            src = ./.;

            nativeBuildInputs = [ pkgs.ruff ];

            buildPhase = ''
              echo "Running Ruff linter checks..."
              ruff check ./src --exclude src/tests
            '';

            installPhase = "mkdir $out";
          };
          basedpyright-types = pkgs.stdenvNoCC.mkDerivation {
            name = "basedpyright-types";
            src = ./.;

            nativeBuildInputs = [
              venvDev
              pkgs.basedpyright
            ];

            buildPhase = ''
              echo "Running Basedpyright type checks..."
              ln -sfn ${venvDev} ./src/.venv
              basedpyright
            '';

            installPhase = "mkdir $out";
          };
          pytest = pkgs.stdenvNoCC.mkDerivation {
            name = "pytest";
            src = ./.;

            nativeBuildInputs = [ venvDev ];

            buildPhase = ''
              echo "Running pytest..."
              export PYTHONPATH=$(pwd)/src
              pytest
            '';

            installPhase = "mkdir $out";
          };
        };
      }
    );
}
