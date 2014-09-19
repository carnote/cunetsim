for i in `seq 137 10 500`
do
	let "nb = $i * $i"
	./Cunetsim -n $nb 
done
