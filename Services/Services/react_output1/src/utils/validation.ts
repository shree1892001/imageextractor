const requiredValidator = (value: string) => value.trim() !== '';
const maxLengthValidator = (value: string, maxLength: number) => value.length <= maxLength;