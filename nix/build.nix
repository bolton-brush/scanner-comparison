{
  stdenvNoCC,
  pythonEnv,
  ...
}:
stdenvNoCC.mkDerivation {
  name = "scanner-comparison";
  src = ../src/scanner_comparison;

  # Pass the production venv into the build environment so we can run collectstatic
  nativeBuildInputs = [ pythonEnv ];

  installPhase = ''
    mkdir -p $out/share/bfd9000_web
    cp -r * .* $out/share/bfd9000_web
  '';
}
