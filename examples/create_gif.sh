#!/usr/bin/env bash

# run and generate the separate gifs
python gans_gif.py

# everything else in here
cd ../resources

# pre
rm -rf combined.gif                                                 # clean previous

# per-gif
magick convert wgan_gp.gif -gravity West -chop 50x0 -gravity East -chop 50x0 wgan_gp.gif   # cut empty sides
magick convert began.gif -gravity West -chop 50x0 -gravity East -chop 50x0 began.gif       # cut empty sides
magick convert wgan_gp.gif -coalesce a-%04d.gif                            # separate frames of wgan_gp.gif
magick convert began.gif -coalesce b-%04d.gif                              # separate frames of began.gif

# combine
for f in a-*.gif; do magick convert ${f} ${f/a/b} +append ${f}; done       # append frames side-by-side
magick convert -loop 0 -delay 60 a-*.gif combined.gif                      # rejoin frames

# post
rm -rf a-* b-*                                                             # cleanup
magick convert combined.gif -fuzz 10% -layers Optimize combined.gif        # reduce size
