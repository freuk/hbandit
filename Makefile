# this file was tested using GNUMAKE >= 4.2.1.

# this is necessary for using multi-line strings as command arguments.
SHELL := $(shell which bash)

# this allows omitting newlines.
.ONESHELL:

# "nix-shell -p" constructs an expression that relies on <nixpkgs> for
# selecting attributes, so we override it.
# https://github.com/NixOS/nix/issues/726#issuecomment-161215255
NIX_PATH := nixpkgs=./.

.PHONY: all
all: hbandit.nix ghcid pre-commit


#generating the vendored cabal file.

.PHONY: ci
ci:
	@nix-shell -p yq -p jq --run bash <<< '
		for jobname in $$(yq -r "keys| .[]" .gitlab-ci.yml); do
			if [ "$$jobname" != "stages" ]; then
				gitlab-runner exec shell "$$jobname"
			fi
		done
	'

ci-%:
	@nix-shell --run bash <<< '
		gitlab-runner exec shell "$*"
	'

.PHONY: ghcid
ghcid: ghcid-hbandit

ghcid-hbandit: hbandit.cabal .hlint.yaml hbandit.nix
	@nix-shell -E '
		with import <nixpkgs> {};
		with haskellPackages;
		shellFor {
			packages = p: [p.hbandit];
			buildInputs = [ghcid cabal-install hlint];
		}
	' --pure --run bash <<< '
		ghcid --command "cabal v2-repl hbandit " \
			--restart=hbandit.cabal \
			--restart=default.nix \
			-l
	'

ghcid-test: hbandit.cabal .hlint.yaml hbandit.nix
	@nix-shell --pure --run bash <<< '
		ghcid --command "cabal v2-repl test " \
			--restart=hbandit.cabal \
			--restart=default.nix \
			-l
	'

.PHONY: pre-commit
pre-commit: ormolu shellcheck README.md

.PHONY: shellcheck
shellcheck:
	@nix-shell --pure -p fd shellcheck --run bash <<< '
		for F in $$(fd -e sh); do
			shellcheck -s bash $$F
		done
	'

.PHONY: hlint
hlint:
	@nix-shell --pure -p hlint --run bash <<< '
		hlint src/ --hint=./.hlint.yaml
	'

.PHONY: ormolu
ormolu:
	@nix-shell --pure -E '
		let pkgs = import <nixpkgs> {};
		in pkgs.mkShell {
			buildInputs = [pkgs.fd pkgs.ormolu];
			shellHook =
				"export LOCALE_ARCHIVE=$${pkgs.glibcLocales}/lib/locale/locale-archive \n" +
				"export LANG=en_US.UTF-8";
		}
	' --run bash <<< '
		RETURN=0
		for F in $$(fd -e hs); do
			ormolu -o -XTypeApplications -o -XPatternSynonyms -m check $$F
			if [ $$? -ne 0 ]; then
				echo "[!] $$F does not pass ormolu format check. Formatting.." >&2
				ormolu -o -XTypeApplications -o -XPatternSynonyms -m inplace $$F
				RETURN=1
			fi
		done
		if [ $$RETURN -ne 0 ]; then exit 1; fi
	'

.PHONY: doc
doc: hbandit.cabal hbandit.nix 
	@nix-shell -E '
		with import <nixpkgs> {};
		with haskellPackages;
		shellFor {
			packages = p: [p.hbandit];
			buildInputs = [cabal-install];
		}
	' --run <<< bash '
		cabal v2-haddock hbandit --haddock-internal
	'

.PHONY:clean
clean:
	rm -rf .build
	rm -rf dist*
	rm -f extras/main.hs
	rm -f hbandit.nix
	rm -f hbandit.cabal
