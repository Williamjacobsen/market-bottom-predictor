{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    buildInputs = with pkgs.python3Packages; [ numpy yfinance pandas matplotlib tqdm ];
}
