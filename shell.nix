with import ./. { };

haskellPackages.shellFor {
  packages = p: [ haskellPackages.hbandit ];
  withHoogle = true;
  buildInputs = [
    ghcid
    haskellPackages.panpipe
    haskellPackages.panhandle
    ormolu
    haskellPackages.hlint
    pandoc
    cabal-install
  ];
  shellHook = ''
    export LOCALE_ARCHIVE=${glibcLocales}/lib/locale/locale-archive
    export LANG=en_US.UTF-8
    export NIX_GHC="${haskellPackages.hbandit.env.NIX_GHC}"
    export NIX_GHCPKG="${haskellPackages.hbandit.env.NIX_GHCPKG}"
    export NIX_GHC_DOCDIR="${haskellPackages.hbandit.env.NIX_GHC_DOCDIR}"
    export NIX_GHC_LIBDIR="${haskellPackages.hbandit.env.NIX_GHC_LIBDIR}"
  '';
}

