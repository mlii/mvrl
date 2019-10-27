for i in `seq 1 25`;
do
  echo worker $i
  # on cloud:
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python extract.py &
  # on macbook for debugging:
  #python extract.py &
  sleep 60.0
done
