github:
    @make html
    @cp -a _build/html/. ../docs
    @cp -a ../other/* ../docs
