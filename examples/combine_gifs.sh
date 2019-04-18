#!/usr/bin/env bash

cd ../resources

rm -rf combined.gif                                                 # clean previous
convert wgan_gp.gif -coalesce a-%04d.gif                            # separate frames of wgan_gp.gif
convert began.gif -coalesce b-%04d.gif                              # separate frames of began.gif
for f in a-*.gif; do convert ${f} ${f/a/b} +append ${f}; done       # append frames side-by-side
convert -loop 0 -delay 60 a-*.gif combined.gif                      # rejoin frames
rm -rf a-* b-*                                                      # cleanup
