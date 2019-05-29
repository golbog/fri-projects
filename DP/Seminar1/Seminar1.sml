Control.Print.printDepth := 100;

datatype expression =  Constant of int |
        Variable of string |
        Operator of string * expression |
        Pair of expression list |
        List of expression list

datatype pattern
  = ConstantP of int
  | ListP of pattern list
  | OperatorP of string * pattern
  | PairP of pattern list
  | VariableP of string
  | Wildcard

exception InvalidExpression of expression
exception InvalidVariable of string

(* 
1. cross (fn : 'a list * 'b list -> ('a * 'b) list) - prejme terko dveh seznamov in vrne njun kartezični produkt 
   (seznam vseh možnih parov elementov). Vrstni red generiranih terk naj bo tak kot je v primeru.
*)
fun cross (a, b) =
    List.foldl (fn (ela, acc) => acc@(List.map (fn elb => (ela, elb)) b)) [] a

(*
2. combinations (fn : 'a list list -> 'a list list) - prejme seznam seznamov in vrne vsa možna zaporedja elementov, 
   pri katerem prvo število izbiramo iz prvega podseznama, drugo iz drugega, ... Vrstni red zaporedij naj bo tak 
   kot je v primeru.
*)
fun combinations [] = []
    | combinations (h::[]) = List.map (fn x => [x]) h
    | combinations (h::t) = (List.concat(List.map(fn x => List.map(fn y => [x] @ y ) (combinations t) ) h))

(*
3. eval (fn : (string * int) list -> expression -> int) - sprejme seznam parov (ime_spremenljivke, vrednost) in izraz. 
   Uporablja currying. Vrne izračunano vrednost izraza. Podpira operatorje +, -, *, /, %, ki predstavljajo 
   seštevanje, odštevanje, množenje, celoštevilsko deljenje ter ostanek pri celoštevilskem deljenju. Evaluacija 
   spremenljivke vrne prvo vrednost iz seznama parov, pri kateri je ime enako imenu spremenljivke. 
*)
fun eval a e =
    let 
        fun find_var (a:(string*int)list, x:string) =
            case a of
                [] => raise InvalidVariable(x)
                | (s,v)::r => if x = s then v else find_var (r,x)
    in
        case e of
            Constant x => x
            | Variable x => find_var(a,x)
            | Operator (x, ex) => (case (x,ex) of
                                    ("+",Pair [e1,e2]) => (eval a e1) + (eval a e2)
                                    | ("+",List l) => foldl op+ 0 (List.map (fn y => eval a y) l)
                                    | ("*",Pair [e1,e2]) => (eval a e1) * (eval a e2)
                                    | ("*",List l) => foldl op* 1 (List.map (fn y => eval a y) l)
                                    | ("-", Pair [e1,e2]) => (eval a e1) - (eval a e2)
                                    | ("/", Pair [e1,e2]) => let val divisor = (eval a e2) in
                                                                if divisor = 0
                                                                then raise InvalidExpression e2
                                                                else (eval a e1) div divisor end
                                    | ("%", Pair [e1,e2]) => let val divisor = (eval a e2) in
                                                                if divisor = 0
                                                                then raise InvalidExpression e
                                                                else (eval a e1) mod divisor end
                                    | _ => raise InvalidExpression ex
                                )
            | _ => raise InvalidExpression e
    end


(* 
4. derivative (fn : expression -> string -> expression) - sprejme izraz in ime spremenljivke po kateri odvajamo. Uporablja currying. Vrne odvod podanega izraza. Podpira naj vsaj izraze, 
   ki vsebujejo poljubno kombinacijo operatorjev +, -, *, / ter konstant in spremenljivk. Predpostavite lahko, da so operatorji vedno 
   podani v enostavni (Pair) obliki. Vrnjenega odvoda ni treba poenostavljati. 
*)
fun derivative e d =
    case e of
        Constant x => Constant 0
        | Variable x => if x = d then Constant 1 else Constant 0
        | Operator ("+", Pair [a1, a2]) => Operator ("+", Pair [derivative a1 d, derivative a2 d])
        | Operator ("-", Pair [a1, a2]) => Operator ("-", Pair [derivative a1 d, derivative a2 d])
        | Operator ("*", Pair [a1, a2]) => Operator ("+", Pair [Operator("*",Pair [(derivative a1 d),a2]), Operator ("*",Pair [a1,(derivative a2 d)])])
        | Operator ("/", Pair [a1, a2]) => Operator ("/", Pair [Operator("-",Pair [Operator ("*", Pair [(derivative a1 d),a2]),Operator("*",Pair [a1,(derivative a2 d)])]), Operator("*",Pair [a2,a2])])
        | _ => raise InvalidExpression(e) 


(*
5. flatten (fn : expression -> expression) - poenostavi podani izraz tako, da ga zapiše kot vsoto produktov. Podpira naj vse izraze, ki 
   vsebujejo poljubno kombinacijo operatorjev + in *. Vsota produktov vsebuje največ dva nivoja operatorjev. Pri tem lahko produkt vsebuje 
   samo konstante in spremenljivke, vsota pa produkte, konstante in spremenljivke.
*)
fun flatten ex = 
    case ex of
        Constant x => Constant x
        | Variable x => Variable x
        | Operator ("*", Pair ls) => flatten (Operator("*", List ls))
        | Operator ("+", Pair ls) => flatten (Operator("+", List ls)) 
        | Operator ("-", Pair ls) => flatten (Operator("-", List ls)) 
        | Operator ("*", List ls) => (let val l = List.map(fn x=> Operator("*", List(x))) (combinations(
                                                    List.foldl(fn (x,acc) => (case flatten x of 
                                                          Operator("+", List a) => a::acc
                                                        | Operator("-", List a) => a::acc
                                                        | Operator("*", List a) => acc@(List.map(fn x => [x]) a)
                                                        | Variable a => [Variable a]::acc
                                                        | Constant a => [Constant a]::acc
                                                        | _ => raise InvalidExpression x)) [] ls))
                                        in 
                                            case l of 
                                                h::[] => h
                                                | _ => Operator("+", List l)
                                        end)
        | Operator ("+", List ls) => Operator("+", List(List.concat(List.map(fn x => (case flatten x of 
                                                         Variable a => [Variable a]
                                                         | Constant a => [Constant a]
                                                         | Operator("+", List a) => a
                                                         | Operator("-", List a) => a
                                                         | Operator("*", List a) => [Operator("*", List a)]
                                                         | _ => raise InvalidExpression x)) ls)))
        | Operator ("-", List([])) => Constant 0
        | Operator ("-", List(h::[])) => h
        | Operator ("-", List ls) => Operator("+", List(List.concat(List.map(fn x => (case flatten x of 
                                                         Variable a => [Constant ~1,Variable a]
                                                         | Constant a => [Constant ~1,Constant a]
                                                         | Operator("-", List a) => a
                                                         | Operator("+", List a) => a
                                                         | Operator("*", List a) => [Operator("*", List a)]
                                                         | _ => raise InvalidExpression x)) ls)))

        | _ => raise InvalidExpression(ex)

(*
6. joinSimilar fn : expression -> expression - poenostavi izraz tako, da združi podobne dele. Vhodni izraz bo v obliki vsote produktov 
   (kot je definirana zgoraj), v enaki obliki naj bo tudi vrnjen izraz.
*)
fun joinSimilar e = 
    let
        fun getHelper f0 f1 f2 f3 e acc= 
            let
                fun tailGetHelper (e, acc) = 
                    case e of
                        Operator ("*",List l) => (case l of
                                    [] => acc
                                    | (Constant x)::t => tailGetHelper(Operator ("*",List t), f0(x,acc))
                                    | (Variable x)::t => tailGetHelper(Operator ("*",List t), f1(x,acc))
                                    | _::t => tailGetHelper(Operator ("*",List t), acc)
                                )
                        | Operator ("*",Pair l) => tailGetHelper (Operator ("*", List l), acc)
                        | Constant x => f2 x
                        | Variable x => f3 x
                        | _ => raise InvalidExpression e
            in
                tailGetHelper(e,acc)
            end
        fun getConstant e = getHelper (fn (x,y) => x*y) (fn (x,y) => y) (fn x => x) (fn _ => 1) e 1
        fun getVariables e = List.map (fn x => Variable x) (ListMergeSort.sort (fn (x,y) => x > y) (getHelper (fn (x,y) => y) (fn (x,y) => x::y) (fn _ => []) (fn x => [x]) e []))
        fun joinSimilarHelper e =
            case e of
                Operator ("+", List []) => []
                | Operator ("+", List l) => let val vars = getVariables(hd l) in 
                                                [Operator ("*", List ((Constant (List.foldl (fn (el, acc) => getConstant(el)+acc) 0 (List.filter (fn x => vars = getVariables x) l)))::vars))]@(joinSimilarHelper (Operator ("+", List (List.filter (fn x => vars <> getVariables x) l))))
                                            end
                | _ => raise InvalidExpression e
    in
        Operator("+", List (joinSimilarHelper e))
    end
    


(*
7. removeEmpty fn : expression -> expression - poenostavi izraz tako, da odstrani odvečne dele (množenja/seštevanja enega elementa, množenja,
   pri katerih je eden izmed členov 0, množenje z ena, prištevanje ničle, odštevanje ničle, deljenje z 1).
*)
fun removeEmpty e =
    case e of
        Constant a => Constant a
        | Variable a => Variable a
        | Operator ("/", Pair [a,b]) => let val b = removeEmpty b in if b = Constant 1 then removeEmpty a else Operator ("/", Pair [removeEmpty a, b]) end
        | Operator ("&", Pair [a,b]) => let val b = removeEmpty b in if b = Constant 1 then Constant 0 else Operator ("&", Pair [removeEmpty a, b]) end
        | Operator ("-", Pair [a,b]) => if removeEmpty b = Constant 0 then removeEmpty a
                                        else Operator ("-", Pair [removeEmpty a,removeEmpty b])
        | Operator ("*", List l) => let val cleaned = List.concat (List.map (fn x => let val emptied = removeEmpty x in if emptied = Constant 1 then [] else [emptied] end) l) in
                                    if List.exists (fn x => x = Constant 0) cleaned then Constant 0
                                    else if cleaned = [] then Constant 1
                                    else if tl cleaned = [] then hd cleaned
                                    else Operator ("*", List cleaned) end
        | Operator ("*", Pair l) => removeEmpty (Operator ("*", List l))
        | Operator ("+", List l) => let val cleaned = List.concat (List.map (fn x => let val emptied = removeEmpty x in if emptied = Constant 0 then [] else [emptied] end) l) in
                                    case cleaned of
                                        [] => Constant 0
                                        | h::[] => h
                                        | _ => Operator ("+", List(cleaned)) end
        | Operator ("+", Pair l) => removeEmpty (Operator ("+", List l))
        | _ => raise InvalidExpression(e)

(*
8. simplify (fn : expression -> expression) - poenostavi izraz tako, da vsebuje kar najmanjše število operandov. Poenostaviti ga mora vsaj toliko, 
   kot je mogoče z uporabo predhodnih funkcij.
*)
fun simplify e =
    removeEmpty(joinSimilar(removeEmpty(flatten e)))
    handle InvalidExpression x => removeEmpty(e)

(*
9.  match (fn : expression * pattern -> (string * expression) list option) - sprejme izraz in vzorec (pattern). V primeru ujemanja vrne SOME seznam vezav, sicer pa NONE.
*)
fun match (e:expression, p:pattern) =
    let 
        fun interleave x [] = [[x]]
            | interleave x (h::t) =
            (x::h::t)::(List.map(fn l => h::l) (interleave x t))
        fun permute nil = [[]]
            | permute (h::t) = List.concat (List.map (fn l => interleave h l) (permute t))
    in 
        case (p,e) of 
            (Wildcard, _) => SOME []
            | (VariableP x, y) => SOME [(x, y)]
            | (ConstantP x, Constant y) => if x = y then SOME [] else NONE
            | (PairP [a1,a2], Pair [b1,b2]) =>  let val t1 = match(b1,a1) val t2 = match(b2,a2) in
                                                if isSome t1 andalso isSome t2 then SOME (valOf(t1)@valOf(t2))
                                                else NONE end
            | (ListP ps, List xs) => let val comb = List.find (fn x => ListPair.foldl(fn (x,y,acc) => isSome(match(x,y)) andalso acc) true (x,ps)) (permute xs) in
                                    if List.length ps = List.length xs andalso isSome(comb)
                                    then SOME (ListPair.foldr(fn (x,y,acc) => (valOf(match(x,y))@acc)) [] (valOf(comb), ps)) 
                                    else NONE end
            | (OperatorP (a,x), Operator (b,y)) => if a=b then match(y,x) else NONE
            | _ => NONE
    end
