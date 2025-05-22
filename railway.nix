{ pkgs }:
pkgs.mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.nodejs_18
  ];
} 