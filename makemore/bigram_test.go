package makemore

import (
	"fmt"
	"slices"
	"testing"
)

func TestRuneToIdx(t *testing.T) {
	testCases := [](struct {
		input  rune
		output int
	}){
		{
			input:  'a',
			output: 0,
		},
		{
			input:  'e',
			output: 4,
		},
		{
			input:  'z',
			output: 25,
		},
		{
			input:  StartOfString,
			output: 26,
		},
		{
			input:  EndOfString,
			output: 27,
		},
	}

	for _, tc := range testCases {
		result := runeToIdx(tc.input)
		if result != tc.output {
			t.Errorf("expected: %d, but got %d for input: %c", tc.output, result, tc.input)
		}
	}
}

func TestToRunePairs(t *testing.T) {
	testCases := [](struct {
		input  string
		output [][2]rune
	}){
		{
			input: frameString("a"),
			output: [][2]rune{
				{StartOfString, 'a'},
				{'a', EndOfString},
			},
		},
		{
			input: frameString("abcd"),
			output: [][2]rune{
				{StartOfString, 'a'},
				{'a', 'b'},
				{'b', 'c'},
				{'c', 'd'},
				{'d', EndOfString},
			},
		},
	}

	for _, tc := range testCases {
		result := toRunePairs(tc.input)
		if !slices.Equal(result, tc.output) {
			t.Errorf("expected: %v, but got %v for input: %s", tc.output, result, tc.input)
		}
	}

}

func TestGenerateProbabilityMatrix(t *testing.T) {
	t.Run("single word", func(t *testing.T) {
		result := GenerateProbabilityMatrix([]string{"aabc"})

		// For "aabc", the bigrams are: (Start→a), (a→a), (a→b), (b→c), (c→End)
		// Total bigrams: 5, so each has probability 1/5 = 0.20

		testCases := []struct {
			name     string
			from     rune
			to       rune
			expected float64
		}{
			{"Start→a", StartOfString, 'a', 0.20},
			{"a→a", 'a', 'a', 0.20},
			{"a→b", 'a', 'b', 0.20},
			{"b→c", 'b', 'c', 0.20},
			{"c→End", 'c', EndOfString, 0.20},
			{"a→End (doesn't exist)", 'a', EndOfString, 0.00},
			{"Start→b (doesn't exist)", StartOfString, 'b', 0.00},
		}

		for _, tc := range testCases {
			prob := result.Get(runeToIdx(tc.from), runeToIdx(tc.to))
			if fmt.Sprintf("%.2f", prob) != fmt.Sprintf("%.2f", tc.expected) {
				t.Errorf("%s: expected %.2f, but got %.2f", tc.name, tc.expected, prob)
			}
		}
	})

	t.Run("multiple words with repeated bigrams", func(t *testing.T) {
		result := GenerateProbabilityMatrix([]string{"ab", "ab"})

		// For "ab" (twice):
		// First "ab": (Start→a), (a→b), (b→End) = 3 bigrams
		// Second "ab": (Start→a), (a→b), (b→End) = 3 bigrams
		// Total: 6 bigrams
		// Frequencies: Start→a=2, a→b=2, b→End=2

		testCases := []struct {
			name     string
			from     rune
			to       rune
			expected float64
		}{
			{"Start→a", StartOfString, 'a', 2.0 / 6.0},
			{"a→b", 'a', 'b', 2.0 / 6.0},
			{"b→End", 'b', EndOfString, 2.0 / 6.0},
			{"a→a (doesn't exist)", 'a', 'a', 0.00},
		}

		for _, tc := range testCases {
			prob := result.Get(runeToIdx(tc.from), runeToIdx(tc.to))
			if fmt.Sprintf("%.4f", prob) != fmt.Sprintf("%.4f", tc.expected) {
				t.Errorf("%s: expected %.4f, but got %.4f", tc.name, tc.expected, prob)
			}
		}
	})

	t.Run("empty strings are skipped", func(t *testing.T) {
		result := GenerateProbabilityMatrix([]string{"a", "", "b"})

		// "a": (Start→a), (a→End) = 2 bigrams
		// "": skipped
		// "b": (Start→b), (b→End) = 2 bigrams
		// Total: 4 bigrams

		testCases := []struct {
			name     string
			from     rune
			to       rune
			expected float64
		}{
			{"Start→a", StartOfString, 'a', 1.0 / 4.0},
			{"a→End", 'a', EndOfString, 1.0 / 4.0},
			{"Start→b", StartOfString, 'b', 1.0 / 4.0},
			{"b→End", 'b', EndOfString, 1.0 / 4.0},
		}

		for _, tc := range testCases {
			prob := result.Get(runeToIdx(tc.from), runeToIdx(tc.to))
			if fmt.Sprintf("%.4f", prob) != fmt.Sprintf("%.4f", tc.expected) {
				t.Errorf("%s: expected %.4f, but got %.4f", tc.name, tc.expected, prob)
			}
		}
	})
}

func TestBuildTrainingExamples(t *testing.T) {
	examples := buildTrainingExamples("./names.txt")
	fmt.Printf("Input:\n%+v\n\n", examples[0].Input)
	fmt.Printf("Output:\n%+v\n\n", examples[0].Output)
}
